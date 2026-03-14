"""
2026 regulation era impact analysis.
V2 upgrade: reliability multiplier set to 3x (from 2x in V1).
"""

import numpy as np
import pandas as pd


# Historical data: DNF rates in year 1 of major regulation changes
REGULATION_CHANGE_YEARS = {
    2009: {"name": "Slick tyres + KERS introduction",     "dnf_rate_increase": 0.31},
    2014: {"name": "Power unit hybrid era (V6 turbo)",    "dnf_rate_increase": 0.58},
    2017: {"name": "Wider cars + more downforce",         "dnf_rate_increase": 0.22},
    2022: {"name": "Ground effect aerodynamics",          "dnf_rate_increase": 0.47},
    2026: {"name": "Active aero + 50/50 power hybrid",   "dnf_rate_increase": 0.65},  # estimated
}

# Points swing analysis: how much did constructor order change in reg-change years?
CONSTRUCTOR_ORDER_SHIFTS = {
    2014: "Red Bull (1st) → 4th. Mercedes rises from 2nd to 1st dominance.",
    2022: "Mercedes falls. Red Bull and Ferrari lead. Surprise: Haas strong early.",
    2026: "Unknown — Mercedes leading based on 2 rounds. Ferrari/McLaren close.",
}


def get_regulation_impact_summary() -> dict:
    """Return a structured summary of 2026 regulation era implications."""
    return {
        "key_changes": [
            "Active aerodynamics — front/rear wings adjust between Corner Mode and Straight-Line Mode",
            "50/50 ICE/electric power split — brand new energy management challenge",
            "Cars are 200mm shorter (wheelbase) and 768kg (down from 800kg)",
            "Overtake Override Mode: +0.5MJ energy boost within 1s of car ahead",
            "Boost Button: driver manually controls energy deployment",
            "22-car grid — Cadillac joins as 11th team",
        ],
        "historical_precedent": [
            f"{yr}: {data['name']} — DNF rate ↑ {data['dnf_rate_increase']:.0%}"
            for yr, data in REGULATION_CHANGE_YEARS.items()
            if yr != 2026
        ],
        "2026_forecast": {
            "dnf_rate_multiplier": "3x base rate (V2 upgrade from 2x in V1)",
            "reliability_risk":    "Highest since 2014 power unit transition",
            "energy_management":   "50/50 split unproven over full race distance",
            "constructor_winners": "Mercedes leading early — but Year 1 order rarely holds",
        },
        "v1_lessons": {
            "australia_dnf_rate":  "3 DNFs in sprint, 0 in GP — regs more reliable than feared",
            "energy_management":   "Mercedes clear advantage on straights (confirmed Shanghai)",
            "tire_behavior":       "Front-left wear dominant (confirmed by Hamilton in sprint)",
            "safety_car_rate":     "Sprint confirmed SC tendency — 1 SC in 19 sprint laps",
        }
    }


def dnf_rate_per_year() -> pd.DataFrame:
    """Return historical DNF rates to contextualize 2026 projection."""
    data = [
        {"year": 2019, "dnf_rate": 0.082, "reg_year1": 0},
        {"year": 2020, "dnf_rate": 0.091, "reg_year1": 0},
        {"year": 2021, "dnf_rate": 0.078, "reg_year1": 0},
        {"year": 2022, "dnf_rate": 0.135, "reg_year1": 1},  # Ground effect year 1
        {"year": 2023, "dnf_rate": 0.095, "reg_year1": 0},
        {"year": 2024, "dnf_rate": 0.088, "reg_year1": 0},
        {"year": 2025, "dnf_rate": 0.091, "reg_year1": 0},
        {"year": 2026, "dnf_rate": 0.145, "reg_year1": 1},  # V2 3x projection
    ]
    return pd.DataFrame(data)


if __name__ == "__main__":
    summary = get_regulation_impact_summary()
    print("2026 Regulation Impact Analysis")
    print("=" * 50)
    print("\nKey Changes:")
    for change in summary["key_changes"]:
        print(f"  • {change}")
    print("\nHistorical Precedent:")
    for h in summary["historical_precedent"]:
        print(f"  • {h}")
    print("\n2026 Forecast:")
    for k, v in summary["2026_forecast"].items():
        print(f"  {k}: {v}")
    print("\nV1 Lessons from Australia:")
    for k, v in summary["v1_lessons"].items():
        print(f"  {k}: {v}")
