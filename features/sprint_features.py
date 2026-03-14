"""
Sprint-derived feature builder — Tier 0 enriched with inferred race signals.
Also computes cross-weekend sprint-to-GP correlation from 2021-2025.
"""

import pandas as pd
import numpy as np
from data.sprint_data import (
    SPRINT_RESULT, SPRINT_QUALI_GRID, SPRINT_AVG_LAP,
    get_sprint_to_gp_likelihood,
)
from data.historical_data import GRID_2026


def compute_sprint_pace_matrix() -> pd.DataFrame:
    """
    Compute derived pace signals from sprint lap times.
    Returns DataFrame with driver pace metrics.
    """
    drivers = list(GRID_2026.keys())
    russell_avg = SPRINT_AVG_LAP["russell"]

    rows = []
    for driver in drivers:
        avg_lap = SPRINT_AVG_LAP.get(driver, 100.5)
        sprint_res = SPRINT_RESULT.get(driver, {})
        sprint_pos = sprint_res.get("sprint_pos", 22)
        is_dnf = sprint_res.get("status", "") == "DNF"
        gap = sprint_res.get("gap_to_winner", None)

        # Per-lap gap (19 sprint laps)
        if gap is not None:
            per_lap_gap = gap / 19.0
        else:
            per_lap_gap = (avg_lap - russell_avg)

        # Pace percentile rank
        all_laps = [SPRINT_AVG_LAP.get(d, 100.5) for d in drivers if SPRINT_RESULT.get(d, {}).get("status") != "DNF"]
        pct_rank = pd.Series(all_laps).rank(pct=True).iloc[
            [d for d in drivers if SPRINT_RESULT.get(d, {}).get("status") != "DNF"].index(driver)
            if driver in [d for d in drivers if SPRINT_RESULT.get(d, {}).get("status") != "DNF"] else 0
        ] if driver in [d for d in drivers if SPRINT_RESULT.get(d, {}).get("status") != "DNF"] else 0.5

        # Inferred long-run capacity (DNF drivers get 0.5 as unknown)
        longrun_estimate = max(0.0, 1.0 - (avg_lap - russell_avg) / 3.0) if not is_dnf else 0.5

        rows.append({
            "driver_id":               driver,
            "sprint_avg_lap_s":        round(avg_lap, 3),
            "sprint_per_lap_gap":      round(per_lap_gap, 4),
            "sprint_longrun_estimate": round(longrun_estimate, 3),
            "sprint_is_dnf":           int(is_dnf),
        })

    return pd.DataFrame(rows).set_index("driver_id")


def compute_sprint_to_gp_likelihoods(drivers: list) -> pd.Series:
    """
    Map each driver's sprint finishing position to their sprint-to-GP likelihood.
    Used in the Bayesian model as evidence update.
    """
    likelihoods = {}
    for driver in drivers:
        sprint_pos = SPRINT_RESULT.get(driver, {}).get("sprint_pos", 22)
        if SPRINT_RESULT.get(driver, {}).get("status") == "DNF":
            sprint_pos = 20
        likelihoods[driver] = get_sprint_to_gp_likelihood(sprint_pos)
    return pd.Series(likelihoods, name="sprint_to_gp_likelihood")


def get_sprint_narrative_flags(drivers: list) -> pd.DataFrame:
    """
    Encode specific sprint race narrative observations as binary flags.
    These are high-level editorial encodings from race analysis.
    """
    # Sprint narrative: key moments
    narratives = {
        "hamilton":   {
            "charge_from_back":         0,
            "led_briefly":              1,   # Led before lap 5
            "energy_management_issue":  1,   # Killed left tyre
            "safety_car_beneficiary":   0,
            "showed_race_pace":         1,
        },
        "russell":    {
            "charge_from_back":         0,
            "led_briefly":              0,
            "energy_management_issue":  0,
            "safety_car_beneficiary":   0,
            "showed_race_pace":         1,   # Dominant winner
        },
        "leclerc":    {
            "charge_from_back":         1,   # P6 → P2
            "led_briefly":              0,
            "energy_management_issue":  0,
            "safety_car_beneficiary":   0,
            "showed_race_pace":         1,
        },
        "antonelli":  {
            "charge_from_back":         1,   # Recovered from penalty
            "led_briefly":              0,
            "energy_management_issue":  0,
            "safety_car_beneficiary":   0,
            "showed_race_pace":         0,   # Penalised for collision
        },
        "verstappen": {
            "charge_from_back":         1,   # P16 → P9
            "led_briefly":              0,
            "energy_management_issue":  0,
            "safety_car_beneficiary":   0,
            "showed_race_pace":         0,   # Net negative result from grid
        },
    }

    rows = []
    for driver in drivers:
        n = narratives.get(driver, {})
        rows.append({
            "driver_id":                    driver,
            "sprint_led_race":              n.get("led_briefly", 0),
            "sprint_charge_from_back":      n.get("charge_from_back", 0),
            "sprint_energy_mgmt_issue":     n.get("energy_management_issue", 0),
            "sprint_sc_beneficiary":        n.get("safety_car_beneficiary", 0),
            "sprint_showed_true_race_pace": n.get("showed_race_pace", 0),
        })
    return pd.DataFrame(rows).set_index("driver_id")


def get_full_sprint_feature_set(drivers: list = None) -> pd.DataFrame:
    """Return complete sprint feature DataFrame combining all sprint sub-modules."""
    if drivers is None:
        drivers = list(GRID_2026.keys())

    pace = compute_sprint_pace_matrix()
    likelihoods = compute_sprint_to_gp_likelihoods(drivers)
    narratives = get_sprint_narrative_flags(drivers)

    combined = pace.join(likelihoods).join(narratives)
    return combined


if __name__ == "__main__":
    drivers = list(GRID_2026.keys())
    df = get_full_sprint_feature_set(drivers)
    print("Sprint Feature Set:")
    print(df.to_string())
