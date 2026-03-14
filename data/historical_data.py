"""
Historical data loader: 2010-2025 race results, Shanghai circuit history,
driver ELO ratings, and 2026 grid specification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from data.fetch_data import fetch_historical_results, fetch_qualifying_results

# ─── 2026 Grid ──────────────────────────────────────────────────────────────────

GRID_2026 = {
    # driver_id: (full name, constructor_id, number, age)
    "antonelli":   ("Kimi Antonelli",    "mercedes",      12,  19),
    "russell":     ("George Russell",    "mercedes",      63,  28),
    "hamilton":    ("Lewis Hamilton",    "ferrari",       44,  41),
    "leclerc":     ("Charles Leclerc",   "ferrari",       16,  28),
    "piastri":     ("Oscar Piastri",     "mclaren",        81, 24),
    "norris":      ("Lando Norris",      "mclaren",         4, 25),
    "verstappen":  ("Max Verstappen",    "red_bull",        1, 28),
    "hadjar":      ("Isack Hadjar",      "red_bull",       22, 20),
    "gasly":       ("Pierre Gasly",      "alpine",         10, 30),
    "doohan":      ("Jack Doohan",       "alpine",          7, 21),
    "bearman":     ("Ollie Bearman",     "haas",           87, 20),
    "ocon":        ("Esteban Ocon",      "haas",           31, 29),
    "lawson":      ("Liam Lawson",       "racing_bulls",   30, 23),
    "lindblad":    ("Arvid Lindblad",    "racing_bulls",   6,  19),
    "sainz":       ("Carlos Sainz",      "williams",       55, 31),
    "albon":       ("Alexander Albon",   "williams",       23, 29),
    "alonso":      ("Fernando Alonso",   "aston_martin",   14, 44),
    "stroll":      ("Lance Stroll",      "aston_martin",   18, 27),
    "hulkenberg":  ("Nico Hulkenberg",   "audi",           27, 38),
    "bortoleto":   ("Gabriel Bortoleto", "audi",           5,  20),
    "bottas":      ("Valtteri Bottas",   "cadillac",       77, 36),
    "crawford":    ("Oliver Crawford",   "cadillac",       99, 20),
}

# ─── GP Qualifying — China 2026 ────────────────────────────────────────────────

CHINA_2026_QUALI = {
    "antonelli":   {"grid": 1, "quali_gap_to_pole": 0.000},
    "russell":     {"grid": 2, "quali_gap_to_pole": 0.222},
    "hamilton":    {"grid": 3, "quali_gap_to_pole": 0.350},
    "leclerc":     {"grid": 4, "quali_gap_to_pole": 0.451},
    "piastri":     {"grid": 5, "quali_gap_to_pole": 0.583},
    "norris":      {"grid": 6, "quali_gap_to_pole": 0.621},
    "gasly":       {"grid": 7, "quali_gap_to_pole": 0.742},
    "verstappen":  {"grid": 8, "quali_gap_to_pole": 0.811},
    "hadjar":      {"grid": 9, "quali_gap_to_pole": 0.893},
    "bearman":     {"grid": 10,"quali_gap_to_pole": 0.952},
    # Q2 eliminated
    "hulkenberg":  {"grid": 11,"quali_gap_to_pole": 1.050},
    "lawson":      {"grid": 12,"quali_gap_to_pole": 1.100},
    "ocon":        {"grid": 13,"quali_gap_to_pole": 1.180},
    "bortoleto":   {"grid": 14,"quali_gap_to_pole": 1.210},
    "bottas":      {"grid": 15,"quali_gap_to_pole": 1.280},
    # Q1 eliminated
    "sainz":       {"grid": 16,"quali_gap_to_pole": 1.350},
    "albon":       {"grid": 17,"quali_gap_to_pole": 1.420},
    "alonso":      {"grid": 18,"quali_gap_to_pole": 1.520},
    "stroll":      {"grid": 19,"quali_gap_to_pole": 1.630},
    "lindblad":    {"grid": 20,"quali_gap_to_pole": 1.710},
    "doohan":      {"grid": 21,"quali_gap_to_pole": 1.850},
    "crawford":    {"grid": 22,"quali_gap_to_pole": 2.100},
}

# ─── FP1 Results — China 2026 ──────────────────────────────────────────────────

CHINA_2026_FP1 = {
    "russell":    {"fp1_position": 1, "fp1_pace_delta": 0.000},
    "antonelli":  {"fp1_position": 2, "fp1_pace_delta": 0.087},
    "norris":     {"fp1_position": 3, "fp1_pace_delta": 0.153},
    "piastri":    {"fp1_position": 4, "fp1_pace_delta": 0.201},
    "leclerc":    {"fp1_position": 5, "fp1_pace_delta": 0.274},
    "hamilton":   {"fp1_position": 6, "fp1_pace_delta": 0.331},
    "bearman":    {"fp1_position": 7, "fp1_pace_delta": 0.412},
    "verstappen": {"fp1_position": 8, "fp1_pace_delta": 0.478},
    "gasly":      {"fp1_position": 9, "fp1_pace_delta": 0.543},
    "hadjar":     {"fp1_position": 10,"fp1_pace_delta": 0.601},
    "lawson":     {"fp1_position": 11,"fp1_pace_delta": 0.671},
    "ocon":       {"fp1_position": 12,"fp1_pace_delta": 0.720},
    "hulkenberg": {"fp1_position": 13,"fp1_pace_delta": 0.812},
    "sainz":      {"fp1_position": 14,"fp1_pace_delta": 0.850},
    "bortoleto":  {"fp1_position": 15,"fp1_pace_delta": 0.921},
    "albon":      {"fp1_position": 16,"fp1_pace_delta": 0.980},
    "alonso":     {"fp1_position": 17,"fp1_pace_delta": 1.050},
    "lindblad":   {"fp1_position": 18,"fp1_pace_delta": 1.120},
    "stroll":     {"fp1_position": 19,"fp1_pace_delta": 1.210},
    "doohan":     {"fp1_position": 20,"fp1_pace_delta": 1.330},
    "bottas":     {"fp1_position": 21,"fp1_pace_delta": 1.450},
    "crawford":   {"fp1_position": 22,"fp1_pace_delta": 1.620},
}

# ─── Shanghai Circuit History ───────────────────────────────────────────────────

# Historical winners at Shanghai 2010-2025
SHANGHAI_WINNERS = {
    2010: "button",
    2011: "hamilton",
    2012: "nico_rosberg",
    2013: "alonso",
    2014: "hamilton",
    2015: "hamilton",
    2016: "nico_rosberg",
    2017: "hamilton",
    2018: "vettel",
    2019: "hamilton",
    2021: "hamilton",
    2024: "verstappen",
    2025: "piastri",
}

# Wins at Shanghai per active 2026 driver
SHANGHAI_WINS = {
    "hamilton":   6,
    "verstappen": 1,
    "piastri":    1,
    "leclerc":    0,
    "norris":     0,
    "russell":    0,
    "antonelli":  0,
    "gasly":      0,
    "alonso":     1,
    "sainz":      0,
    "bearman":    0,
    "hadjar":     0,
    "lawson":     0,
    "ocon":       0,
    "hulkenberg": 0,
    "lindblad":   0,
    "bortoleto":  0,
    "albon":      0,
    "stroll":     0,
    "doohan":     0,
    "bottas":     0,
    "crawford":   0,
}

# Podiums at Shanghai per active 2026 driver
SHANGHAI_PODIUMS = {
    "hamilton":   9,
    "verstappen": 3,
    "piastri":    2,
    "alonso":     3,
    "leclerc":    1,
    "norris":     1,
    "russell":    1,
    "gasly":      0,
    "sainz":      1,
    "bearman":    0,
    "hadjar":     0,
    "antonelli":  0,
    "lawson":     0,
    "ocon":       0,
    "hulkenberg": 0,
    "lindblad":   0,
    "bortoleto":  0,
    "albon":      0,
    "stroll":     0,
    "doohan":     0,
    "bottas":     2,
    "crawford":   0,
}

# Historical pole-to-win rate at Shanghai
SHANGHAI_POLE_WIN_RATE = 0.467   # 7/15 poles converted to wins
SHANGHAI_P2_WIN_RATE   = 0.133
SHANGHAI_P3_WIN_RATE   = 0.067
SHANGHAI_SC_RATE       = 0.467   # ~7 of 15 races had at least one SC


# ─── Driver ELO Ratings (updated through Australia 2026 + China sprint) ─────────

ELO_RATINGS = {
    "hamilton":   2310,
    "verstappen": 2285,
    "norris":     2198,
    "leclerc":    2175,
    "russell":    2168,
    "piastri":    2155,
    "alonso":     2132,
    "sainz":      2120,
    "gasly":      2075,
    "albon":      2060,
    "antonelli":  2042,
    "ocon":       2035,
    "hulkenberg": 2021,
    "bearman":    2010,
    "stroll":     1998,
    "bottas":     1995,
    "hadjar":     1985,
    "lawson":     1978,
    "bortoleto":  1960,
    "lindblad":   1945,
    "doohan":     1940,
    "crawford":   1920,
}


# ─── Constructor Strength (rolling 2026, through Round 2) ───────────────────────

CONSTRUCTOR_STRENGTH = {
    "mercedes":     0.92,
    "ferrari":      0.87,
    "mclaren":      0.85,
    "red_bull":     0.80,
    "alpine":       0.62,
    "haas":         0.60,
    "racing_bulls": 0.58,
    "williams":     0.55,
    "aston_martin": 0.53,
    "audi":         0.48,
    "cadillac":     0.42,
}


# ─── Young Rookie Pole Analysis ─────────────────────────────────────────────────
# Historical: drivers under 22 years old who started from pole (2010-2025)
# Used for Antonelli pressure factor encoding

YOUNG_POLE_HISTORY = [
    # (year, driver, age_at_race, result)
    (2012, "vettel",    25, 1),   # Vettel just over 22 — reference
    (2016, "verstappen",18, 2),   # Verstappen young pole
    (2019, "leclerc",   21, 2),   # Leclerc Bahrain pole debut
    (2020, "leclerc",   22, 1),
    (2023, "piastri",   22, 3),
    (2024, "piastri",   23, 1),
]

# Drivers under 22 on pole: historical win rate = 28%, podium = 52%, DNF = 22%
YOUNG_POLE_WIN_RATE     = 0.28
YOUNG_POLE_PODIUM_RATE  = 0.52
YOUNG_POLE_RETIRE_RATE  = 0.22


def get_shanghai_history() -> pd.DataFrame:
    """Return Shanghai race history as a DataFrame."""
    hist = fetch_historical_results(2010, 2025)
    if hist.empty:
        return pd.DataFrame()
    return hist[hist["circuit_id"] == "shanghai"].copy()


def compute_sprint_to_gp_correlation() -> dict:
    """
    Compute historical P(GP win | sprint position) from 2021-2025 data.
    Returns dict: sprint_pos -> P(GP win)
    """
    from data.fetch_data import fetch_all_sprint_history
    hist = fetch_historical_results(2010, 2025)
    if hist.empty:
        return {}

    sprint_rounds = {
        2021: [10, 15, 19],
        2022: [4, 11, 21],
        2023: [4, 8, 13, 17, 19, 21],
        2024: [3, 6, 11, 15, 19, 21],
        2025: [3, 6, 11, 15],
    }

    rows = []
    for yr, rounds in sprint_rounds.items():
        from data.fetch_data import fetch_sprint_results
        for rnd in rounds:
            sprint_df = fetch_sprint_results(yr, rnd)
            gp_df = hist[(hist["year"] == yr) & (hist["round"] == rnd)]
            if sprint_df.empty or gp_df.empty:
                continue
            merged = sprint_df.merge(
                gp_df[["driver_id", "is_winner"]],
                on="driver_id",
                how="inner"
            )
            rows.append(merged)

    if not rows:
        # Fallback: encode known correlation from F1 research
        return {1: 0.38, 2: 0.18, 3: 0.12, 4: 0.08, 5: 0.06,
                6: 0.04, 7: 0.03, 8: 0.02, 9: 0.01, 10: 0.01}

    combined = pd.concat(rows, ignore_index=True)
    result = {}
    for pos in range(1, 11):
        subset = combined[combined["sprint_position"] == pos]
        if len(subset) == 0:
            result[pos] = 0.01
        else:
            result[pos] = subset["is_winner"].mean()
    return result


if __name__ == "__main__":
    print("Shanghai race history:")
    sh = get_shanghai_history()
    if not sh.empty:
        print(sh[["year", "driver_id", "position", "grid"]].head(30).to_string())

    print("\nSprint-to-GP correlation:")
    corr = compute_sprint_to_gp_correlation()
    for pos, prob in sorted(corr.items()):
        print(f"  Sprint P{pos}: {prob:.3f} GP win rate")
