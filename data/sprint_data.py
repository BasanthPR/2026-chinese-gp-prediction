"""
Sprint-specific data pipeline (NEW V2).
Encodes the 2026 China sprint result as Tier 0 features for the GP prediction.
"""

import pandas as pd
import numpy as np

# ─── 2026 China Sprint Weekend Data ────────────────────────────────────────────

# Sprint qualifying grid (SQ3)
SPRINT_QUALI_GRID = {
    "russell":    1,
    "antonelli":  2,
    "norris":     3,
    "hamilton":   4,
    "piastri":    5,
    "leclerc":    6,
    "gasly":      7,
    "verstappen": 8,
    "bearman":    9,
    "hadjar":     10,
    "lawson":     11,
    "ocon":       12,
    "hulkenberg": 13,
    "sainz":      14,
    "bortoleto":  15,
    "albon":      16,
    "alonso":     17,
    "stroll":     18,
    "bottas":     19,
    "lindblad":   20,
    "doohan":     21,
    "crawford":   22,
}

# Sprint race result (19 laps, Saturday)
SPRINT_RESULT = {
    "russell":    {"sprint_pos": 1,  "gap_to_winner": 0.000, "status": "Finished"},
    "leclerc":    {"sprint_pos": 2,  "gap_to_winner": 0.674, "status": "Finished"},
    "hamilton":   {"sprint_pos": 3,  "gap_to_winner": 2.554, "status": "Finished"},
    "norris":     {"sprint_pos": 4,  "gap_to_winner": 4.433, "status": "Finished"},
    "antonelli":  {"sprint_pos": 5,  "gap_to_winner": 5.688, "status": "Finished (+10s penalty)"},
    "piastri":    {"sprint_pos": 6,  "gap_to_winner": 6.809, "status": "Finished"},
    "lawson":     {"sprint_pos": 7,  "gap_to_winner": 10.900,"status": "Finished"},
    "bearman":    {"sprint_pos": 8,  "gap_to_winner": 11.271,"status": "Finished"},
    "verstappen": {"sprint_pos": 9,  "gap_to_winner": 11.619,"status": "Finished"},
    "ocon":       {"sprint_pos": 10, "gap_to_winner": 13.887,"status": "Finished"},
    "hadjar":     {"sprint_pos": 11, "gap_to_winner": 16.200,"status": "Finished"},
    "gasly":      {"sprint_pos": 12, "gap_to_winner": 17.100,"status": "Finished"},
    "sainz":      {"sprint_pos": 13, "gap_to_winner": 18.500,"status": "Finished"},
    "albon":      {"sprint_pos": 14, "gap_to_winner": 20.100,"status": "Finished"},
    "bortoleto":  {"sprint_pos": 15, "gap_to_winner": 21.300,"status": "Finished"},
    "stroll":     {"sprint_pos": 16, "gap_to_winner": 22.800,"status": "Finished"},
    "alonso":     {"sprint_pos": 17, "gap_to_winner": 24.100,"status": "Finished"},
    "doohan":     {"sprint_pos": 18, "gap_to_winner": 25.600,"status": "Finished"},
    "crawford":   {"sprint_pos": 19, "gap_to_winner": 27.200,"status": "Finished"},
    "hulkenberg": {"sprint_pos": 99, "gap_to_winner": None,  "status": "DNF"},
    "bottas":     {"sprint_pos": 99, "gap_to_winner": None,  "status": "DNF"},
    "lindblad":   {"sprint_pos": 99, "gap_to_winner": None,  "status": "DNF"},
}

# Sprint fastest lap: Leclerc
SPRINT_FASTEST_LAP_DRIVER = "leclerc"

# Sprint average lap time (seconds) — 19 laps over the sprint
# Russell baseline, others derived from gap / 19 laps
SPRINT_AVG_LAP = {
    "russell":    98.20,
    "leclerc":    98.24,
    "hamilton":   98.33,
    "norris":     98.43,
    "antonelli":  98.50,   # true pace ~98.24 without penalty
    "piastri":    98.56,
    "lawson":     98.77,
    "bearman":    98.79,
    "verstappen": 98.81,
    "ocon":       98.93,
    "hadjar":     99.05,
    "gasly":      99.10,
    "sainz":      99.17,
    "albon":      99.28,
    "bortoleto":  99.34,
    "stroll":     99.40,
    "alonso":     99.47,
    "doohan":     99.55,
    "crawford":   99.63,
    "hulkenberg": 99.20,   # estimated before DNF
    "bottas":     99.35,
    "lindblad":   99.50,
}

# ─── Tier 0 Feature Flags ──────────────────────────────────────────────────────

# sprint_tire_stress_flag: 1 if driver showed tire degradation signs in sprint
SPRINT_TIRE_STRESS = {
    "hamilton":   1,   # Explicitly said "killed my left tyre"
    "verstappen": 1,   # Pushed hard through the field after P16 drop
    "bearman":    1,   # Had oversteer incidents in FP1, aggressive style
    "leclerc":    0,
    "russell":    0,
    "norris":     0,
    "piastri":    0,
    "antonelli":  0,
    "gasly":      0,
    "hadjar":     0,
    "lawson":     0,
    "ocon":       0,
    "hulkenberg": 0,
    "sainz":      0,
    "bortoleto":  0,
    "albon":      0,
    "alonso":     0,
    "stroll":     0,
    "doohan":     0,
    "bottas":     0,
    "lindblad":   0,
    "crawford":   0,
}

# sprint_reliability_flag: 1 if car/driver showed reliability or incident concern
SPRINT_RELIABILITY_FLAG = {
    "russell":    1,   # Q3 power/gear issue
    "antonelli":  1,   # Sprint collision penalty, crash tendency
    "hulkenberg": 1,   # DNF in sprint
    "bottas":     1,   # DNF in sprint
    "lindblad":   1,   # DNF in sprint
    "hamilton":   0,
    "leclerc":    0,
    "norris":     0,
    "piastri":    0,
    "verstappen": 0,
    "gasly":      0,
    "bearman":    0,
    "hadjar":     0,
    "lawson":     0,
    "ocon":       0,
    "sainz":      0,
    "albon":      0,
    "alonso":     0,
    "stroll":     0,
    "bortoleto":  0,
    "doohan":     0,
    "crawford":   0,
}


def build_sprint_features(drivers: list) -> pd.DataFrame:
    """
    Build Tier 0 sprint features for all 22 drivers.
    Returns a DataFrame indexed by driver_id.
    """
    rows = []
    russell_avg = SPRINT_AVG_LAP["russell"]

    for driver in drivers:
        sprint_res = SPRINT_RESULT.get(driver, {})
        sprint_pos = sprint_res.get("sprint_pos", 22)
        sprint_status = sprint_res.get("status", "Finished")
        sq_grid = SPRINT_QUALI_GRID.get(driver, 22)

        # DNF drivers get sprint_pos = 20 (penalized but not as bad as last)
        if sprint_status == "DNF":
            sprint_pos_encoded = 20
        else:
            sprint_pos_encoded = sprint_pos

        # Pace gap to Russell (seconds per lap)
        avg_lap = SPRINT_AVG_LAP.get(driver, 100.0)
        sprint_pace_gap = avg_lap - russell_avg

        # Positions gained in sprint
        sprint_pos_gained = sq_grid - sprint_pos_encoded

        rows.append({
            "driver_id":               driver,
            "sprint_finishing_position": sprint_pos_encoded,
            "sprint_pace_gap_to_winner": round(sprint_pace_gap, 3),
            "sprint_positions_gained":   sprint_pos_gained,
            "sprint_tire_stress_flag":   SPRINT_TIRE_STRESS.get(driver, 0),
            "sprint_reliability_flag":   SPRINT_RELIABILITY_FLAG.get(driver, 0),
            "sprint_fastest_lap_holder": 1 if driver == SPRINT_FASTEST_LAP_DRIVER else 0,
            "sprint_dnf":                1 if sprint_status == "DNF" else 0,
            "sprint_quali_pos":          sq_grid,
        })

    df = pd.DataFrame(rows).set_index("driver_id")
    return df


def get_sprint_to_gp_likelihood(sprint_pos: int) -> float:
    """
    Return the likelihood boost/penalty for a Bayesian update
    based on sprint finishing position.
    Derived from 2021-2025 historical sprint weekends.
    """
    # P(GP win | sprint position) likelihood table
    likelihood_table = {
        1:  0.38,   # Sprint winner has ~38% elevated GP win probability
        2:  0.18,
        3:  0.12,
        4:  0.08,
        5:  0.06,
        6:  0.05,
        7:  0.04,
        8:  0.03,
        9:  0.02,
        10: 0.02,
        11: 0.015,
        12: 0.013,
        13: 0.010,
        14: 0.008,
        15: 0.007,
        16: 0.006,
        17: 0.005,
        18: 0.004,
        19: 0.003,
        20: 0.002,  # DNF-encoded
    }
    return likelihood_table.get(sprint_pos, 0.002)


if __name__ == "__main__":
    from data.historical_data import GRID_2026
    drivers = list(GRID_2026.keys())
    df = build_sprint_features(drivers)
    print("Sprint Tier 0 Features:")
    print(df[["sprint_finishing_position",
              "sprint_pace_gap_to_winner",
              "sprint_positions_gained",
              "sprint_tire_stress_flag",
              "sprint_reliability_flag"]].to_string())
