"""
Feature configuration, constants, and team/driver metadata for V2.
"""

# ─── Feature Tiers ─────────────────────────────────────────────────────────────

TIER0_FEATURES = [
    "sprint_finishing_position",
    "sprint_pace_gap_to_winner",
    "sprint_positions_gained",
    "sprint_tire_stress_flag",
    "sprint_reliability_flag",
    "sprint_fastest_lap_holder",
    "sprint_dnf",
]

TIER1_FEATURES = [
    "grid_position",
    "quali_gap_to_pole",
    "fp1_pace_delta",
    "fp1_position",
    "fp1_to_quali_divergence",
    "sprint_quali_vs_gp_quali_delta",
]

TIER2_FEATURES = [
    "constructor_strength",
    "driver_elo_rating",
    "driver_elo_normalized",
    "driver_circuit_wins",
    "driver_circuit_podiums",
    "teammate_quali_gap",
    "driver_new_team_flag",
]

TIER3_FEATURES = [
    "is_new_regulation_year",
    "reliability_risk_multiplier",
    "circuit_energy_profile",
    "circuit_overtaking_difficulty",
    "safety_car_probability",
    "wet_race_probability",
    "active_aero_era",
]

TIER4_FEATURES = [
    "ferrari_flip_flop_wing",
    "antonelli_pressure_factor",
    "energy_depletion_risk",
    "rookie_pole_variance",
]

ALL_FEATURES = TIER0_FEATURES + TIER1_FEATURES + TIER2_FEATURES + TIER3_FEATURES + TIER4_FEATURES

# ─── XGBoost Training Feature Set ──────────────────────────────────────────────
# Subset that can be derived from historical records (for training)

HISTORICAL_FEATURES = [
    # Always available
    "grid_position",
    "constructor_strength",
    "driver_elo_rating",
    "driver_elo_normalized",
    "driver_circuit_wins",
    "driver_circuit_podiums",
    "is_new_regulation_year",
    "reliability_risk_multiplier",
    # Available for most recent years
    "fp1_position",
    "fp1_to_quali_divergence",
    "quali_gap_to_pole",
    # Sprint features (2021-2025 only)
    "sprint_finishing_position",
    "sprint_positions_gained",
    "sprint_pace_gap_to_winner",
]

TARGET = "is_winner"

# ─── Shanghai Circuit Constants ────────────────────────────────────────────────

CIRCUIT = {
    "name":                "Shanghai International Circuit",
    "circuit_id":          "shanghai",
    "laps":                56,
    "lap_length_km":       5.451,
    "total_distance_km":   305.066,
    "sectors":             3,
    "corners":             16,
    "back_straight_km":    1.175,
    "drs_zones":           2,
    "energy_profile":      0.88,   # 0-1 scale, high = more energy intensive
    "overtaking_difficulty": 0.45, # 0-1 scale, mid-range
    "sc_probability":      0.56,   # Adjusted for sprint evidence
    "pole_win_rate":       0.467,
}

# ─── Tire Window Constants (sprint-confirmed) ───────────────────────────────────

TIRE_WINDOWS = {
    "soft":   {"min": 15, "max": 18, "deg_rate": 0.065},    # s/lap degradation
    "medium": {"min": 25, "max": 30, "deg_rate": 0.038},
    "hard":   {"min": 35, "max": 45, "deg_rate": 0.022},
}

# Turn 1 aggressive starters: 15% higher front-left degradation
AGGRESSIVE_STARTER_DEG_MULTIPLIER = 1.15
AGGRESSIVE_STARTERS = ["hamilton", "verstappen", "bearman"]

# ─── Energy Management Rankings (2026-specific) ────────────────────────────────
# Straight-line pace gain vs base (seconds/lap on Shanghai back straight)

ENERGY_MANAGEMENT = {
    "mercedes":     0.000,  # Benchmark — best energy recovery
    "ferrari":     -0.050,
    "mclaren":     -0.080,
    "red_bull":    -0.120,
    "alpine":      -0.150,
    "haas":        -0.160,
    "racing_bulls":-0.175,
    "williams":    -0.190,
    "aston_martin":-0.200,
    "audi":        -0.230,
    "cadillac":    -0.270,
}

# ─── DNF Rate Configuration (V2 — 3x multiplier) ──────────────────────────────

BASE_DNF_RATE_PER_LAP  = 0.005     # 0.5% per lap (historical 2022-2025 avg)
REG_YEAR1_MULTIPLIER   = 3.0       # V2 upgrade from 2x
# Additional per-driver multipliers
DRIVER_DNF_MULTIPLIERS = {
    "russell":    0.5,    # Q3 power/gear issue this weekend
    "antonelli":  0.3,    # Sprint collision, crash tendency
    "hulkenberg": 0.2,    # Sprint DNF
    "bottas":     0.2,    # Sprint DNF
    "lindblad":   0.3,    # Sprint DNF + rookie
    "crawford":   0.2,    # Rookie
    "doohan":     0.15,
}

# ─── Pit Strategy Configuration ────────────────────────────────────────────────

PIT_STRATEGY = {
    "pit_loss_seconds":       22.0,
    "undercut_window_start":  22,
    "undercut_window_end":    26,
    "second_stop_window":     38,
    "overcut_risk":           0.15,  # Ferrari historical overcut tendency
}

# Ferrari pit strategy penalty (historical: missed VSC pit in Australia 2026)
FERRARI_PIT_STRATEGY_RISK = 0.20   # 20% chance of suboptimal pit call

# ─── Post-Ensemble Adjustments ─────────────────────────────────────────────────

RELIABILITY_PENALTIES = {
    "russell":    0.92,   # Q3 car issue
    "antonelli":  0.92,   # Sprint reliability concern
}

FERRARI_WING_CI_EXPANSION = 0.05   # ±5% confidence interval expansion

WET_RACE_PROBABILITY = 0.25   # Blend weight for wet ensemble

# Grid position cap: P8+ drivers capped at 5% unless sprint_positions_gained > 3
GRID_CAP_THRESHOLD  = 8
GRID_CAP_MAX_PROB   = 0.05
GRID_CAP_OVERRIDE_SPRINT_GAIN = 3
