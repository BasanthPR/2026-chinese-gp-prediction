"""
Weighted ensemble model V2.
Weights: MC=45%, XGBoost=30%, Bayesian=25%.
Applies post-ensemble reliability penalties and wet-race blending.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from data.historical_data import GRID_2026, CHINA_2026_QUALI
from data.sprint_data import SPRINT_RESULT
from features.config import (
    RELIABILITY_PENALTIES, FERRARI_WING_CI_EXPANSION,
    WET_RACE_PROBABILITY, GRID_CAP_THRESHOLD, GRID_CAP_MAX_PROB,
    GRID_CAP_OVERRIDE_SPRINT_GAIN,
)
from features.engineering import build_feature_matrix

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

DRIVERS = list(GRID_2026.keys())

# Ensemble weights (V2)
W_XGB  = 0.30
W_MC   = 0.45
W_BAYES = 0.25


def compute_dry_ensemble(
    xgb_probs: pd.Series,
    mc_probs: pd.Series,
    bayes_probs: pd.Series,
) -> pd.Series:
    """Weighted blend of three models (dry race scenario)."""
    # Align on driver index
    idx = pd.Index(DRIVERS)
    xgb   = xgb_probs.reindex(idx).fillna(0)
    mc    = mc_probs.reindex(idx).fillna(0)
    bayes = bayes_probs.reindex(idx).fillna(0)

    # Normalize each to sum to 1 before blending
    xgb   /= xgb.sum()   if xgb.sum()   > 0 else 1
    mc    /= mc.sum()    if mc.sum()    > 0 else 1
    bayes /= bayes.sum() if bayes.sum() > 0 else 1

    ensemble = W_XGB * xgb + W_MC * mc + W_BAYES * bayes
    return ensemble / ensemble.sum()


def compute_wet_ensemble(
    xgb_probs: pd.Series,
    mc_probs: pd.Series,
    bayes_probs: pd.Series,
) -> pd.Series:
    """
    Wet-race boosted ensemble.
    Hamilton and Verstappen receive wet-race boost; rookies penalized.
    """
    from data.weather import WET_RACE_DRIVER_BOOST
    dry = compute_dry_ensemble(xgb_probs, mc_probs, bayes_probs)

    wet_adjusted = dry.copy()
    for driver in DRIVERS:
        boost = WET_RACE_DRIVER_BOOST.get(driver, 1.0)
        wet_adjusted[driver] *= boost

    return wet_adjusted / wet_adjusted.sum()


def apply_post_ensemble_adjustments(
    ensemble: pd.Series,
    feature_matrix: pd.DataFrame,
) -> pd.Series:
    """
    Apply post-blend corrections:
    1. Reliability penalties for Russell (Q3 issue) and Antonelli (crash tendency)
    2. Grid position cap: P8+ drivers capped at 5% unless sprint gain > 3
    3. Ferrari wing uncertainty: widen CI (affects confidence, not mean)
    """
    adjusted = ensemble.copy()

    # 1. Reliability penalties
    for driver, penalty in RELIABILITY_PENALTIES.items():
        if driver in adjusted.index:
            adjusted[driver] *= penalty

    # 2. Grid position cap
    for driver in DRIVERS:
        grid_pos = CHINA_2026_QUALI.get(driver, {"grid": 22})["grid"]
        sprint_gain = SPRINT_RESULT.get(driver, {}).get("sprint_pos", 22)
        sq_grid = feature_matrix.loc[driver, "sprint_quali_pos"] if driver in feature_matrix.index else 22
        sprint_pos_gained = sq_grid - (sprint_gain if sprint_gain != 99 else 20)

        if grid_pos >= GRID_CAP_THRESHOLD and sprint_pos_gained <= GRID_CAP_OVERRIDE_SPRINT_GAIN:
            adjusted[driver] = min(adjusted[driver], GRID_CAP_MAX_PROB)

    # Re-normalize
    total = adjusted.sum()
    if total > 0:
        adjusted = adjusted / total

    return adjusted


def blend_dry_wet(dry_probs: pd.Series, wet_probs: pd.Series) -> pd.Series:
    """Blend dry and wet ensembles by rain probability."""
    p_wet = WET_RACE_PROBABILITY       # 0.25
    p_dry = 1.0 - p_wet

    blended = p_dry * dry_probs + p_wet * wet_probs
    return blended / blended.sum()


def run_ensemble(
    xgb_probs: pd.Series,
    mc_probs: pd.Series,
    bayes_probs: pd.Series,
) -> pd.DataFrame:
    """
    Full ensemble pipeline.
    Returns DataFrame with per-model + ensemble probabilities.
    """
    print("\n[Ensemble] Blending XGBoost + Monte Carlo + Bayesian...")
    print(f"  Weights: XGBoost={W_XGB:.0%}  MC={W_MC:.0%}  Bayes={W_BAYES:.0%}")

    fm = build_feature_matrix()

    # Dry and wet ensembles
    dry  = compute_dry_ensemble(xgb_probs, mc_probs, bayes_probs)
    wet  = compute_wet_ensemble(xgb_probs, mc_probs, bayes_probs)

    # Apply post-ensemble adjustments to dry
    dry_adj = apply_post_ensemble_adjustments(dry, fm)

    # Weather blend
    final = blend_dry_wet(dry_adj, wet)

    print(f"  Rain blend: {WET_RACE_PROBABILITY:.0%} wet / {1-WET_RACE_PROBABILITY:.0%} dry")

    # Build output DataFrame
    idx = pd.Index(DRIVERS)
    result_df = pd.DataFrame({
        "xgb_prob":      xgb_probs.reindex(idx).fillna(0),
        "mc_prob":       mc_probs.reindex(idx).fillna(0),
        "bayes_prob":    bayes_probs.reindex(idx).fillna(0),
        "dry_ensemble":  dry_adj.reindex(idx).fillna(0),
        "wet_ensemble":  wet.reindex(idx).fillna(0),
        "final_ensemble":final.reindex(idx).fillna(0),
    })

    # Add grid and sprint position for reference
    result_df["grid_position"]    = result_df.index.map(
        lambda d: CHINA_2026_QUALI.get(d, {"grid": 22})["grid"])
    result_df["sprint_position"]  = result_df.index.map(
        lambda d: SPRINT_RESULT.get(d, {}).get("sprint_pos", 22))

    # Add metadata from GRID_2026
    result_df["full_name"]   = result_df.index.map(lambda d: GRID_2026[d][0])
    result_df["constructor"] = result_df.index.map(lambda d: GRID_2026[d][1])

    # Sort by final ensemble
    result_df = result_df.sort_values("final_ensemble", ascending=False)

    print("\n[Ensemble] Final Top 10:")
    for driver, row in result_df.head(10).iterrows():
        print(f"  P{int(row['grid_position']):2d} {row['full_name']:20s} "
              f"| Sprint P{int(row['sprint_position']):2d} "
              f"| XGB {row['xgb_prob']:.3f} "
              f"| MC {row['mc_prob']:.3f} "
              f"| Bayes {row['bayes_prob']:.3f} "
              f"| Ensemble {row['final_ensemble']:.3f}")

    # Save
    result_df.to_parquet(OUTPUT_DIR / "ensemble_results.parquet")
    result_df.to_csv(OUTPUT_DIR / "ensemble_results.csv")
    print(f"\n  Saved to {OUTPUT_DIR}/ensemble_results.parquet")

    return result_df


def compute_scenario_matrix(
    xgb_probs: pd.Series,
    mc_probs: pd.Series,
    bayes_probs: pd.Series,
) -> pd.DataFrame:
    """
    Compute 2x2 scenario matrix: (dry/wet) × (SC/no SC).
    Returns DataFrame for visualization.
    """
    from data.weather import WET_RACE_DRIVER_BOOST

    dry_no_sc  = compute_dry_ensemble(xgb_probs, mc_probs, bayes_probs)
    dry_sc     = _apply_sc_boost(dry_no_sc)
    wet_no_sc  = compute_wet_ensemble(xgb_probs, mc_probs, bayes_probs)
    wet_sc     = _apply_sc_boost(wet_no_sc)

    top3 = dry_no_sc.sort_values(ascending=False).head(5).index

    rows = []
    for driver in top3:
        rows.append({
            "driver":      GRID_2026[driver][0],
            "dry_no_sc":   dry_no_sc.get(driver, 0),
            "dry_sc":      dry_sc.get(driver, 0),
            "wet_no_sc":   wet_no_sc.get(driver, 0),
            "wet_sc":      wet_sc.get(driver, 0),
        })
    return pd.DataFrame(rows)


def _apply_sc_boost(probs: pd.Series) -> pd.Series:
    """Safety car equalises field — boosts mid-grid runners, hurts leaders."""
    adjusted = probs.copy()
    for driver in DRIVERS:
        grid_pos = CHINA_2026_QUALI.get(driver, {"grid": 22})["grid"]
        if grid_pos <= 3:
            adjusted[driver] *= 0.82   # Leader loses some advantage under SC
        elif grid_pos <= 8:
            adjusted[driver] *= 1.25   # Mid-pack gains
        elif grid_pos <= 12:
            adjusted[driver] *= 1.40   # Back of top-10 gains most
    return adjusted / adjusted.sum()


if __name__ == "__main__":
    # Quick test with synthetic inputs
    from models.xgboost_model import run_xgboost
    from models.monte_carlo_sim import run_monte_carlo_model
    from models.bayesian_model import run_bayesian_model

    xgb   = run_xgboost()
    mc    = run_monte_carlo_model()
    bayes, ci = run_bayesian_model()

    results = run_ensemble(xgb, mc, bayes)
    print("\nFull ensemble saved.")
