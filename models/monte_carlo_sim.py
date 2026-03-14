"""
Monte Carlo Race Simulator V2 — sprint-calibrated, 10,000 iterations.
56 laps at Shanghai International Circuit.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from data.historical_data import GRID_2026
from data.sprint_data import SPRINT_AVG_LAP, SPRINT_RESULT, SPRINT_TIRE_STRESS
from data.weather import sample_rain_onset, get_weather_scenario, WET_RACE_DRIVER_BOOST
from features.config import (
    CIRCUIT, TIRE_WINDOWS, AGGRESSIVE_STARTERS, AGGRESSIVE_STARTER_DEG_MULTIPLIER,
    ENERGY_MANAGEMENT, DRIVER_DNF_MULTIPLIERS, BASE_DNF_RATE_PER_LAP,
    REG_YEAR1_MULTIPLIER, PIT_STRATEGY, FERRARI_PIT_STRATEGY_RISK,
)
from data.historical_data import CONSTRUCTOR_STRENGTH

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

N_LAPS    = CIRCUIT["laps"]     # 56
N_SIMS    = 10_000
DRIVERS   = list(GRID_2026.keys())
TEAMS     = {d: GRID_2026[d][1] for d in DRIVERS}

# ─── Base lap time matrix (sprint-calibrated) ────────────────────────────────────

# Russell = 0 baseline; all others relative to Russell's sprint pace
# These are per-lap time DIFFERENCES (seconds) — positive = slower
BASE_PACE_DELTA = {d: SPRINT_AVG_LAP.get(d, 100.0) - SPRINT_AVG_LAP["russell"] for d in DRIVERS}

# Pace uncertainty — year 1 of new regs: ±0.3s/lap (V1 was ±0.2)
PACE_SIGMA = 0.30

# ─── DNF rate per lap ─────────────────────────────────────────────────────────

def get_dnf_rate(driver: str) -> float:
    base = BASE_DNF_RATE_PER_LAP * REG_YEAR1_MULTIPLIER
    extra = DRIVER_DNF_MULTIPLIERS.get(driver, 0.0) * BASE_DNF_RATE_PER_LAP
    return base + extra


# ─── Tire model ───────────────────────────────────────────────────────────────

class TireStint:
    """Represents a single tire stint."""
    def __init__(self, compound: str, start_lap: int, driver: str):
        self.compound  = compound
        self.start_lap = start_lap
        self.laps_done = 0
        self.driver    = driver
        self.window    = TIRE_WINDOWS[compound]
        self.deg_rate  = self.window["deg_rate"]
        # Aggressive starters have higher FL degradation
        if driver in AGGRESSIVE_STARTERS or SPRINT_TIRE_STRESS.get(driver, 0):
            self.deg_rate *= AGGRESSIVE_STARTER_DEG_MULTIPLIER

    def lap_delta(self) -> float:
        """Returns additional seconds due to tire wear this lap."""
        if self.laps_done <= 3:
            return 0.0    # New tire grace period
        excess = max(0, self.laps_done - self.window["min"])
        return self.deg_rate * excess

    def is_optimal_window(self) -> bool:
        return self.laps_done <= self.window["max"]

    def step(self):
        self.laps_done += 1


def choose_strategy(driver: str, grid_pos: int) -> list:
    """
    Return a list of (pit_lap, next_compound) tuples for a 2-stop strategy.
    Staggered by grid position to simulate real-world strategy calls.
    """
    base_stop1 = PIT_STRATEGY["undercut_window_start"]
    base_stop2 = PIT_STRATEGY["second_stop_window"]

    # Vary by grid position: frontrunners stop later (track position), backmarkers stop early
    offset = (grid_pos - 1) * 0.4
    stop1 = int(np.clip(base_stop1 + np.random.randint(-2, 3) - offset * 0.5, 15, 30))
    stop2 = int(np.clip(base_stop2 + np.random.randint(-3, 4), 34, 46))

    return [(stop1, "medium"), (stop2, "hard")]


# ─── Single race simulation ────────────────────────────────────────────────────

def simulate_race(rain_onset: int, sim_id: int) -> dict:
    """Simulate one 56-lap race. Returns dict of {driver: finishing_position}."""
    weather = get_weather_scenario(rain_onset)
    is_wet  = weather["is_wet"]
    sc_prob_per_lap = weather["sc_prob_per_lap"]

    # Initialise per-driver state
    state = {}
    for driver in DRIVERS:
        grid = GRID_2026[driver][0]   # Full name (unused here, use CHINA_2026_QUALI)
        from data.historical_data import CHINA_2026_QUALI
        grid_pos = CHINA_2026_QUALI.get(driver, {"grid": 22})["grid"]
        team = TEAMS[driver]
        energy_adv = ENERGY_MANAGEMENT.get(team, -0.2)

        # Sample this driver's race pace for this simulation (noise around sprint-calibrated base)
        pace_noise = np.random.normal(0, PACE_SIGMA)
        base_pace = BASE_PACE_DELTA[driver] + pace_noise

        # Wet race pace adjustment
        if is_wet:
            wet_boost = WET_RACE_DRIVER_BOOST.get(driver, 1.0)
            # Better wet drivers get relatively faster vs field
            base_pace *= (2.0 - wet_boost)

        # Starting tire: soft for top 10, medium for P11+
        start_compound = "soft" if grid_pos <= 10 else "medium"
        strategy = choose_strategy(driver, grid_pos)

        state[driver] = {
            "position":      grid_pos,
            "lap_time_base": base_pace,
            "energy_adv":    energy_adv,
            "tire":          TireStint(start_compound, 0, driver),
            "strategy":      strategy,
            "pit_stop_idx":  0,
            "total_time":    0.0,
            "is_dnf":        False,
            "laps_completed": 0,
            "dnf_rate":      get_dnf_rate(driver),
            "team":          team,
        }

    # Safety car state
    sc_active      = False
    sc_laps_remaining = 0
    sc_occurred    = False

    # Lap-by-lap simulation
    for lap in range(1, N_LAPS + 1):
        # ── Safety car logic ───────────────────────────────────────────────────
        if sc_active:
            sc_laps_remaining -= 1
            if sc_laps_remaining <= 0:
                sc_active = False
        else:
            if np.random.random() < sc_prob_per_lap:
                sc_active = True
                sc_laps_remaining = int(np.random.uniform(3, 7))
                sc_occurred = True
                # Force pit for leaders under safety car (opportunity window)
                for driver, s in state.items():
                    if not s["is_dnf"] and s["pit_stop_idx"] < len(s["strategy"]):
                        planned_lap = s["strategy"][s["pit_stop_idx"]][0]
                        if abs(lap - planned_lap) <= 3:
                            s["strategy"][s["pit_stop_idx"]] = (lap, s["strategy"][s["pit_stop_idx"]][1])

        # ── Rain onset ─────────────────────────────────────────────────────────
        if is_wet and lap == rain_onset:
            # All drivers pit to switch to wet tires
            for driver, s in state.items():
                if not s["is_dnf"]:
                    s["tire"] = TireStint("medium", lap, driver)
                    s["total_time"] += PIT_STRATEGY["pit_loss_seconds"]

        # ── Per-driver lap ─────────────────────────────────────────────────────
        for driver, s in state.items():
            if s["is_dnf"]:
                continue

            # DNF check
            if np.random.random() < s["dnf_rate"]:
                s["is_dnf"] = True
                s["laps_completed"] = lap
                continue

            # Pit stop check
            if s["pit_stop_idx"] < len(s["strategy"]):
                pit_lap, next_compound = s["strategy"][s["pit_stop_idx"]]
                if lap == pit_lap:
                    s["tire"] = TireStint(next_compound, lap, driver)
                    s["pit_stop_idx"] += 1
                    pit_loss = PIT_STRATEGY["pit_loss_seconds"]
                    # Ferrari pit strategy risk
                    if s["team"] == "ferrari" and np.random.random() < FERRARI_PIT_STRATEGY_RISK:
                        pit_loss += np.random.uniform(2, 8)   # Suboptimal stop
                    s["total_time"] += pit_loss
                    if sc_active:
                        s["total_time"] -= 15.0   # SC pit benefit

            # Base lap time
            lap_time = 98.20 + s["lap_time_base"]

            # Tire degradation
            tire_delta = s["tire"].lap_delta()
            lap_time += tire_delta
            s["tire"].step()

            # Energy management advantage on straights
            lap_time += s["energy_adv"]

            # SC lap: all pace equalised
            if sc_active:
                lap_time = 110.0 + np.random.normal(0, 0.5)

            # Fuel correction: cars get lighter → ~0.03s/lap faster each lap
            lap_time -= lap * 0.028

            # Hamilton energy depletion risk: pace ceiling in laps 30+
            if driver == "hamilton" and SPRINT_TIRE_STRESS.get("hamilton", 0) and lap > 30:
                lap_time += np.random.uniform(0, 0.3)

            s["total_time"] += lap_time
            s["laps_completed"] = lap

    # ── Determine finishing order ───────────────────────────────────────────────
    finishers = [(d, s["total_time"]) for d, s in state.items() if not s["is_dnf"]]
    dnfs      = [d for d, s in state.items() if s["is_dnf"]]

    finishers.sort(key=lambda x: x[1])
    result = {}
    for pos, (driver, _) in enumerate(finishers, 1):
        result[driver] = pos
    for i, driver in enumerate(dnfs):
        result[driver] = len(finishers) + 1 + i

    return result


# ─── Full 10,000 simulation run ────────────────────────────────────────────────

def run_monte_carlo(n_sims: int = N_SIMS) -> tuple:
    """
    Run n_sims race simulations.
    Returns (win_counts dict, all_results array, sc_count, wet_count).
    """
    print(f"  Running {n_sims:,} simulations...")
    win_counts      = defaultdict(int)
    podium_counts   = defaultdict(int)
    dnf_counts      = defaultdict(int)
    sc_count        = 0
    wet_count       = 0
    position_sums   = defaultdict(int)
    all_results     = []

    rain_onsets = sample_rain_onset(n_sims, N_LAPS)

    for sim_id in range(n_sims):
        rain_onset = int(rain_onsets[sim_id])
        if rain_onset >= 0:
            wet_count += 1

        result = simulate_race(rain_onset, sim_id)
        all_results.append(result)

        for driver, pos in result.items():
            position_sums[driver] += pos
            if pos == 1:
                win_counts[driver] += 1
            if pos <= 3:
                podium_counts[driver] += 1
            if pos > len(DRIVERS) - 3:
                dnf_counts[driver] += 1

        if sim_id % 1000 == 0 and sim_id > 0:
            print(f"    {sim_id:,}/{n_sims:,} complete...")

    print(f"  Wet races: {wet_count}/{n_sims} ({wet_count/n_sims*100:.1f}%)")

    # Normalize to probabilities
    win_probs = {d: win_counts[d] / n_sims for d in DRIVERS}
    return win_probs, all_results, dnf_counts, podium_counts


def run_monte_carlo_model() -> pd.Series:
    """Full Monte Carlo pipeline."""
    print("\n[Monte Carlo] Sprint-calibrated 10,000-race simulation...")
    win_probs, all_results, dnf_counts, podium_counts = run_monte_carlo()

    probs = pd.Series(win_probs, name="mc_prob")
    probs = probs / probs.sum()   # Normalize

    print("\n[Monte Carlo] Top 5:")
    for driver, prob in probs.sort_values(ascending=False).head(5).items():
        dnf_rate = dnf_counts[driver] / N_SIMS
        print(f"  {driver:15s}: {prob:.3f}  (DNF rate: {dnf_rate:.1%})")

    # Save podium probabilities
    podium_probs = {d: podium_counts[d] / N_SIMS for d in DRIVERS}
    out = pd.DataFrame({
        "win_prob":    win_probs,
        "podium_prob": podium_probs,
        "dnf_prob":    {d: dnf_counts[d] / N_SIMS for d in DRIVERS},
    })
    out.to_parquet(OUTPUT_DIR / "mc_results.parquet")

    # Save raw results for visualization
    results_df = pd.DataFrame(all_results)
    results_df.to_parquet(OUTPUT_DIR / "mc_raw_results.parquet")

    return probs


if __name__ == "__main__":
    probs = run_monte_carlo_model()
    print("\nFull Monte Carlo probability ranking:")
    for driver, prob in probs.sort_values(ascending=False).items():
        print(f"  {driver:15s}: {prob:.4f} ({prob*100:.1f}%)")
