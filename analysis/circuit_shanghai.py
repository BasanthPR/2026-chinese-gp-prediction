"""
Shanghai International Circuit historical analysis.
"""

import numpy as np
import pandas as pd
from data.historical_data import (
    SHANGHAI_WINNERS, SHANGHAI_WINS, SHANGHAI_PODIUMS,
    SHANGHAI_POLE_WIN_RATE, SHANGHAI_SC_RATE,
)


def get_circuit_profile() -> dict:
    """Return structured circuit characteristics."""
    return {
        "general": {
            "name":           "Shanghai International Circuit",
            "location":       "Shanghai, China",
            "length_km":      5.451,
            "laps":           56,
            "corners":        16,
            "drs_zones":      2,
            "lap_record_s":   91.862,  # Michael Schumacher, 2004 (old spec)
            "pole_time_2025": "1:30.641",
        },
        "sectors": {
            "sector_1": "Long spiraling Turn 1-2-3 complex. Front-left destroyer. "
                        "Tests aero under sustained high-g loading.",
            "sector_2": "Back straight (1.175km). Maximum speed energy test. "
                        "Turn 13-14 hairpin: primary overtaking zone.",
            "sector_3": "Technical fast chicanes. Turn 9 (Russell passed Hamilton here in sprint). "
                        "High-speed exits test traction.",
        },
        "strategic": {
            "typical_stops":      2,
            "primary_compound":   "Soft → Medium → Hard",
            "safety_car_rate":    f"{SHANGHAI_SC_RATE:.1%}",
            "pole_to_win_rate":   f"{SHANGHAI_POLE_WIN_RATE:.1%}",
            "lead_lap_overtakes": "Medium difficulty — DRS + Overtake Override Mode help",
        },
        "2026_specific": {
            "energy_demand":         "HIGH — longest back straight + T1 complex demands peak battery",
            "active_aero_impact":    "Turn 9/14 are key overtake mode activation zones",
            "tire_characteristics":  "Front-left dominant wear (confirmed sprint). "
                                     "Abrasive surface accelerates degradation.",
        }
    }


def get_championship_standings_pre_china() -> pd.DataFrame:
    """
    Approximate 2026 driver standings after Australia GP (Round 1) + Sprint pts.
    Used for championship implications chart.
    """
    # Sprint: 8-7-6-5-4-3-2-1 pts for P1-P8
    # GP: 25-18-15-12-10-8-6-4-2-1 pts for P1-P10
    # Australia GP (V1 predicted winner: Russell won, Antonelli P2, Leclerc P3)
    standings = {
        "russell":    25 + 8,   # Australia win + sprint win
        "antonelli":  18 + 4,   # P2 + sprint P5
        "leclerc":    15 + 7,   # P3 + sprint P2
        "hamilton":   12 + 6,   # P4 + sprint P3
        "norris":     10 + 5,   # P5 + sprint P4
        "piastri":    8  + 3,   # P6 + sprint P6
        "lawson":     6  + 2,   # P7 + sprint P7
        "bearman":    4  + 1,   # P8 + sprint P8
        "verstappen": 2  + 0,   # P9 + sprint P9
        "hadjar":     1  + 0,
        "ocon":       0  + 0,
        "gasly":      0  + 0,
        "sainz":      0  + 0,
        "albon":      0  + 0,
        "alonso":     0  + 0,
        "stroll":     0  + 0,
        "hulkenberg": 0  + 0,
        "bortoleto":  0  + 0,
        "bottas":     0  + 0,
        "lindblad":   0  + 0,
        "doohan":     0  + 0,
        "crawford":   0  + 0,
    }
    from data.historical_data import GRID_2026
    df = pd.DataFrame([
        {"driver_id": k, "full_name": GRID_2026[k][0],
         "constructor": GRID_2026[k][1], "points": v}
        for k, v in standings.items()
    ]).sort_values("points", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


GP_POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]


def project_standings_after_china(
    pre_china_standings: pd.DataFrame,
    predicted_finish_order: list,
) -> pd.DataFrame:
    """
    Project standings after China GP based on predicted finish order.
    predicted_finish_order: list of driver_ids from P1 to P22.
    """
    df = pre_china_standings.set_index("driver_id").copy()
    for pos, driver in enumerate(predicted_finish_order[:10]):
        if driver in df.index:
            df.loc[driver, "points"] += GP_POINTS[pos]
    return df.sort_values("points", ascending=False).reset_index()


if __name__ == "__main__":
    profile = get_circuit_profile()
    print("Shanghai Circuit Profile:")
    for section, data in profile.items():
        print(f"\n{section.upper()}:")
        for k, v in data.items():
            print(f"  {k}: {v}")

    standings = get_championship_standings_pre_china()
    print("\nPre-China Standings (after Australia GP + sprints):")
    print(standings[["rank", "full_name", "constructor", "points"]].head(10).to_string(index=False))
