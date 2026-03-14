"""
Full feature engineering pipeline for 2026 China GP V2.
Builds all 4 tiers of features for all 22 drivers.
"""

import pandas as pd
import numpy as np
from data.historical_data import (
    GRID_2026, CHINA_2026_QUALI, CHINA_2026_FP1,
    ELO_RATINGS, CONSTRUCTOR_STRENGTH, SHANGHAI_WINS, SHANGHAI_PODIUMS,
    YOUNG_POLE_WIN_RATE, YOUNG_POLE_RETIRE_RATE,
)
from data.sprint_data import build_sprint_features, SPRINT_QUALI_GRID
from features.config import (
    CIRCUIT, ENERGY_MANAGEMENT, DRIVER_DNF_MULTIPLIERS,
    BASE_DNF_RATE_PER_LAP, REG_YEAR1_MULTIPLIER,
    AGGRESSIVE_STARTERS, FERRARI_PIT_STRATEGY_RISK,
)


def build_feature_matrix() -> pd.DataFrame:
    """
    Build the complete Tier 0-4 feature matrix for all 22 drivers.
    Returns a DataFrame with driver_id as index.
    """
    drivers = list(GRID_2026.keys())

    # ── Tier 0: Sprint Features ─────────────────────────────────────────────────
    tier0 = build_sprint_features(drivers)

    rows = []
    for driver in drivers:
        info = GRID_2026[driver]
        full_name, constructor, number, age = info

        quali = CHINA_2026_QUALI.get(driver, {"grid": 22, "quali_gap_to_pole": 2.5})
        fp1   = CHINA_2026_FP1.get(driver, {"fp1_position": 22, "fp1_pace_delta": 2.0})
        t0    = tier0.loc[driver]

        grid_pos  = quali["grid"]
        fp1_pos   = fp1["fp1_position"]
        sq_pos    = SPRINT_QUALI_GRID.get(driver, 22)

        # ── Tier 1: Qualifying & Practice ──────────────────────────────────────
        fp1_to_quali_div        = fp1_pos - grid_pos
        sprint_quali_gp_delta   = sq_pos - grid_pos

        # ── Tier 2: Driver & Constructor ───────────────────────────────────────
        elo       = ELO_RATINGS.get(driver, 1900)
        elo_norm  = (elo - 1900) / 400   # Normalize 0-1 ish

        # Teammate qualifying gap
        if constructor == "mercedes":
            teammates = {"antonelli": "russell", "russell": "antonelli"}
        elif constructor == "ferrari":
            teammates = {"hamilton": "leclerc", "leclerc": "hamilton"}
        elif constructor == "mclaren":
            teammates = {"norris": "piastri", "piastri": "norris"}
        elif constructor == "red_bull":
            teammates = {"verstappen": "hadjar", "hadjar": "verstappen"}
        elif constructor == "alpine":
            teammates = {"gasly": "doohan", "doohan": "gasly"}
        elif constructor == "haas":
            teammates = {"bearman": "ocon", "ocon": "bearman"}
        elif constructor == "racing_bulls":
            teammates = {"lawson": "lindblad", "lindblad": "lawson"}
        elif constructor == "williams":
            teammates = {"sainz": "albon", "albon": "sainz"}
        elif constructor == "aston_martin":
            teammates = {"alonso": "stroll", "stroll": "alonso"}
        elif constructor == "audi":
            teammates = {"hulkenberg": "bortoleto", "bortoleto": "hulkenberg"}
        elif constructor == "cadillac":
            teammates = {"bottas": "crawford", "crawford": "bottas"}
        else:
            teammates = {}

        tm = teammates.get(driver)
        if tm:
            tm_gap_pole = CHINA_2026_QUALI.get(tm, {"quali_gap_to_pole": 2.5})["quali_gap_to_pole"]
            my_gap_pole = quali["quali_gap_to_pole"]
            teammate_quali_gap = tm_gap_pole - my_gap_pole  # positive = I beat teammate
        else:
            teammate_quali_gap = 0.0

        # New team flag
        new_team_map = {
            "hamilton":  0,   # Ferrari year 2 (2026)
            "antonelli": 1,   # Mercedes rookie
            "bearman":   1,   # Haas rookie
            "hadjar":    1,   # Red Bull rookie
            "lindblad":  1,   # Racing Bulls rookie
            "bortoleto": 1,   # Audi rookie
            "crawford":  1,   # Cadillac rookie
            "doohan":    1,   # Alpine (part season 2025, but new in 2026)
            "lawson":    0,   # Racing Bulls/Red Bull line year 2
        }
        new_team_flag = new_team_map.get(driver, 0)

        # ── Tier 3: Regulation Era & Track ─────────────────────────────────────
        base_dnf = BASE_DNF_RATE_PER_LAP * REG_YEAR1_MULTIPLIER
        extra_dnf = DRIVER_DNF_MULTIPLIERS.get(driver, 0.0)
        reliability_risk = base_dnf + (extra_dnf * BASE_DNF_RATE_PER_LAP)

        # ── Tier 4: Weekend-Specific ────────────────────────────────────────────
        ferrari_flip_flop = 1 if constructor == "ferrari" else 0

        antonelli_pressure = 0.0
        if driver == "antonelli":
            # Antonelli on pole at age 19 — historic moment, higher variance
            # Win rate from young pole history: 28%, retire rate: 22%
            antonelli_pressure = YOUNG_POLE_RETIRE_RATE - YOUNG_POLE_WIN_RATE  # net pressure penalty

        energy_depletion = 1 if driver == "hamilton" and t0["sprint_tire_stress_flag"] == 1 else 0

        rows.append({
            "driver_id":                     driver,
            "full_name":                     full_name,
            "constructor":                   constructor,
            "age":                           age,
            # Tier 0
            "sprint_finishing_position":     t0["sprint_finishing_position"],
            "sprint_pace_gap_to_winner":     t0["sprint_pace_gap_to_winner"],
            "sprint_positions_gained":       t0["sprint_positions_gained"],
            "sprint_tire_stress_flag":       t0["sprint_tire_stress_flag"],
            "sprint_reliability_flag":       t0["sprint_reliability_flag"],
            "sprint_fastest_lap_holder":     t0["sprint_fastest_lap_holder"],
            "sprint_dnf":                    t0["sprint_dnf"],
            "sprint_quali_pos":              t0["sprint_quali_pos"],
            # Tier 1
            "grid_position":                 grid_pos,
            "quali_gap_to_pole":             quali["quali_gap_to_pole"],
            "fp1_pace_delta":                fp1["fp1_pace_delta"],
            "fp1_position":                  fp1_pos,
            "fp1_to_quali_divergence":       fp1_to_quali_div,
            "sprint_quali_vs_gp_quali_delta":sprint_quali_gp_delta,
            # Tier 2
            "constructor_strength":          CONSTRUCTOR_STRENGTH.get(constructor, 0.5),
            "driver_elo_rating":             elo,
            "driver_elo_normalized":         round(elo_norm, 4),
            "driver_circuit_wins":           SHANGHAI_WINS.get(driver, 0),
            "driver_circuit_podiums":        SHANGHAI_PODIUMS.get(driver, 0),
            "teammate_quali_gap":            round(teammate_quali_gap, 3),
            "driver_new_team_flag":          new_team_flag,
            # Tier 3
            "is_new_regulation_year":        1,
            "reliability_risk_multiplier":   round(reliability_risk, 4),
            "circuit_energy_profile":        CIRCUIT["energy_profile"],
            "circuit_overtaking_difficulty": CIRCUIT["overtaking_difficulty"],
            "safety_car_probability":        CIRCUIT["sc_probability"],
            "wet_race_probability":          0.25,
            "active_aero_era":               1,
            "energy_mgmt_advantage":         ENERGY_MANAGEMENT.get(constructor, -0.2),
            # Tier 4
            "ferrari_flip_flop_wing":        ferrari_flip_flop,
            "antonelli_pressure_factor":     round(antonelli_pressure, 3),
            "energy_depletion_risk":         energy_depletion,
            "rookie_pole_variance":          1 if driver == "antonelli" else 0,
        })

    df = pd.DataFrame(rows).set_index("driver_id")
    return df


def build_historical_feature_matrix(race_results: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from historical race data for XGBoost training.
    Handles missing features (e.g., sprint data only available 2021+).
    """
    if race_results.empty:
        return pd.DataFrame()

    df = race_results.copy()

    # Constructor strength: map from historical constructor names
    constructor_map = {
        "mercedes": "mercedes", "ferrari": "ferrari", "mclaren": "mclaren",
        "red_bull": "red_bull", "renault": "alpine", "alpine": "alpine",
        "haas": "haas", "alphatauri": "racing_bulls", "toro_rosso": "racing_bulls",
        "racing_bulls": "racing_bulls", "williams": "williams",
        "force_india": "aston_martin", "racing_point": "aston_martin",
        "aston_martin": "aston_martin", "sauber": "audi", "kick_sauber": "audi",
        "lotus_f1": "racing_bulls", "caterham": "cadillac", "manor": "cadillac",
        "red_bull_junior": "racing_bulls",
    }
    df["constructor_std"] = df["constructor_id"].map(constructor_map).fillna("unknown")
    df["constructor_strength"] = df["constructor_std"].map(CONSTRUCTOR_STRENGTH).fillna(0.45)

    # Is new regulation year
    reg_change_years = {2009, 2014, 2017, 2022, 2026}
    df["is_new_regulation_year"] = df["year"].isin(reg_change_years).astype(int)

    # Reliability risk (simplified for historical)
    df["reliability_risk_multiplier"] = np.where(
        df["is_new_regulation_year"] == 1,
        BASE_DNF_RATE_PER_LAP * REG_YEAR1_MULTIPLIER,
        BASE_DNF_RATE_PER_LAP
    )

    # Grid position
    df["grid_position"] = df["grid"].clip(1, 22)

    # Shanghai-specific circuit history (simplified via is_winner per circuit)
    df["driver_circuit_wins"] = 0   # Would be computed via cumulative sum in full pipeline

    # Sprint features: fill with median/default when not available
    df["sprint_finishing_position"] = df.get("sprint_position", pd.Series(11, index=df.index)).fillna(11)
    df["sprint_positions_gained"]   = 0
    df["sprint_pace_gap_to_winner"] = 0.5

    # ELO: use position-based proxy for historical data
    df["driver_elo_rating"]     = 2100 - (df["grid_position"] * 10)
    df["driver_elo_normalized"] = (df["driver_elo_rating"] - 1900) / 400

    # Quali gap: proxy via grid position
    df["quali_gap_to_pole"] = (df["grid_position"] - 1) * 0.12

    # FP1 features: estimate from grid with noise
    df["fp1_position"]            = df["grid_position"]
    df["fp1_to_quali_divergence"] = 0
    df["fp1_pace_delta"]          = df["quali_gap_to_pole"]

    # Target
    df["is_winner"] = df["is_winner"].astype(int)

    return df


if __name__ == "__main__":
    fm = build_feature_matrix()
    print("Feature matrix shape:", fm.shape)
    print("\nTop 10 drivers by grid position:")
    top10 = fm.sort_values("grid_position").head(10)
    cols = ["full_name", "grid_position", "sprint_finishing_position",
            "sprint_pace_gap_to_winner", "fp1_to_quali_divergence",
            "driver_circuit_wins", "reliability_risk_multiplier"]
    print(top10[cols].to_string())
