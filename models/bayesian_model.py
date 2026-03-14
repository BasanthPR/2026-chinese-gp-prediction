"""
Bayesian inference model V2 — three-stage Beta-Binomial update.
Stage 1: Historical prior (Shanghai 2010-2025)
Stage 2: Sprint evidence update (NEW V2)
Stage 3: GP qualifying grid update
"""

import numpy as np
import pandas as pd
from scipy import stats

from data.historical_data import (
    SHANGHAI_WINS, SHANGHAI_PODIUMS, GRID_2026, CHINA_2026_QUALI,
    YOUNG_POLE_WIN_RATE, YOUNG_POLE_RETIRE_RATE,
    SHANGHAI_POLE_WIN_RATE, SHANGHAI_P2_WIN_RATE, SHANGHAI_P3_WIN_RATE,
    ELO_RATINGS, CONSTRUCTOR_STRENGTH,
)
from data.sprint_data import SPRINT_RESULT, get_sprint_to_gp_likelihood
from features.config import RELIABILITY_PENALTIES


DRIVERS = list(GRID_2026.keys())

# ─── Stage 1: Historical Prior ─────────────────────────────────────────────────

def compute_historical_prior() -> pd.Series:
    """
    Beta-Binomial prior based on Shanghai race history (2010-2025).
    alpha = wins + constructor_strength_pseudo_counts
    beta  = starts - wins + (1 - constructor_strength) * pseudo_counts
    """
    # Approximate number of starts at Shanghai per driver
    SHANGHAI_STARTS = {
        "hamilton":   10, "verstappen": 4,  "alonso":     4,
        "leclerc":    4,  "norris":     3,  "piastri":    3,
        "russell":    3,  "sainz":      4,  "gasly":      4,
        "albon":      3,  "bottas":     8,  "ocon":       3,
        "hulkenberg": 4,  "stroll":     3,  "antonelli":  1,
        "bearman":    1,  "hadjar":     1,  "lawson":     2,
        "lindblad":   0,  "bortoleto":  0,  "doohan":     1,
        "crawford":   0,
    }

    priors = {}
    pseudo = 2.0   # Pseudo-count strength (stronger = more conservative)

    for driver in DRIVERS:
        wins   = SHANGHAI_WINS.get(driver, 0)
        starts = max(SHANGHAI_STARTS.get(driver, 1), 1)
        c_str  = CONSTRUCTOR_STRENGTH.get(GRID_2026[driver][1], 0.5)
        elo    = ELO_RATINGS.get(driver, 2000)
        elo_norm = (elo - 1900) / 400  # Roughly 0-1

        # Alpha: wins + constructor/ELO-weighted pseudo-counts
        alpha = wins + pseudo * c_str * elo_norm
        # Beta: non-wins + inverse pseudo-counts
        beta  = (starts - wins) + pseudo * (1 - c_str)

        # Mean of Beta distribution
        priors[driver] = alpha / (alpha + beta)

    # Normalize to sum to 1
    total = sum(priors.values())
    return pd.Series({d: v / total for d, v in priors.items()}, name="bayes_prior")


# ─── Stage 2: Sprint Evidence Update ──────────────────────────────────────────

def sprint_update(prior: pd.Series) -> pd.Series:
    """
    Update prior using sprint result as likelihood evidence.
    Uses historical sprint-to-GP win correlation from 2021-2025.
    """
    likelihoods = {}
    for driver in DRIVERS:
        sprint_pos = SPRINT_RESULT.get(driver, {}).get("sprint_pos", 22)
        if SPRINT_RESULT.get(driver, {}).get("status") == "DNF":
            sprint_pos = 20
        likelihoods[driver] = get_sprint_to_gp_likelihood(sprint_pos)

    # Bayesian update: posterior ∝ prior × likelihood
    posterior = {}
    for driver in DRIVERS:
        posterior[driver] = prior[driver] * likelihoods[driver]

    # Normalize
    total = sum(posterior.values())
    return pd.Series({d: v / total for d, v in posterior.items()}, name="bayes_sprint_posterior")


# ─── Stage 3: GP Qualifying Grid Update ────────────────────────────────────────

def grid_update(posterior: pd.Series) -> pd.Series:
    """
    Further update using GP qualifying grid position.
    Applies historical Shanghai pole/grid win rate as likelihood.
    """
    # Historical P(win | grid_pos) at Shanghai
    grid_likelihood_table = {
        1:  SHANGHAI_POLE_WIN_RATE,   # 0.467
        2:  SHANGHAI_P2_WIN_RATE,     # 0.133
        3:  SHANGHAI_P3_WIN_RATE,     # 0.067
        4:  0.053,
        5:  0.040,
        6:  0.027,
        7:  0.020,
        8:  0.013,
        9:  0.007,
        10: 0.007,
    }

    def grid_likelihood(grid_pos: int) -> float:
        if grid_pos <= 10:
            return grid_likelihood_table.get(grid_pos, 0.007)
        return max(0.002, 0.007 - (grid_pos - 10) * 0.0005)

    updated = {}
    for driver in DRIVERS:
        grid_pos = CHINA_2026_QUALI.get(driver, {"grid": 22})["grid"]
        lk = grid_likelihood(grid_pos)
        updated[driver] = posterior[driver] * lk

    # Antonelli special case: young pole variance adjustment
    if "antonelli" in updated:
        updated["antonelli"] *= YOUNG_POLE_WIN_RATE / SHANGHAI_POLE_WIN_RATE

    # Normalize
    total = sum(updated.values())
    updated_series = pd.Series({d: v / total for d, v in updated.items()},
                               name="bayes_grid_posterior")

    return updated_series


# ─── Credible Intervals ─────────────────────────────────────────────────────────

def compute_credible_intervals(posterior: pd.Series, n_samples: int = 10_000) -> pd.DataFrame:
    """
    Compute 90% credible intervals for each driver's win probability
    using the Dirichlet-Multinomial model.
    """
    # Use posterior as concentration parameters for Dirichlet distribution
    # Scale up to get meaningful alpha values
    alphas = (posterior * 100).clip(0.1)

    rows = []
    samples = np.random.dirichlet(alphas.values, size=n_samples)

    for i, driver in enumerate(posterior.index):
        driver_samples = samples[:, i]
        ci_lo = np.percentile(driver_samples, 5)
        ci_hi = np.percentile(driver_samples, 95)
        rows.append({
            "driver_id": driver,
            "mean":      posterior[driver],
            "ci_lo_90":  ci_lo,
            "ci_hi_90":  ci_hi,
        })
    return pd.DataFrame(rows).set_index("driver_id")


# ─── Full Bayesian Pipeline ─────────────────────────────────────────────────────

def run_bayesian_model() -> tuple:
    """
    Three-stage Bayesian update.
    Returns (final_posterior Series, credible_interval DataFrame).
    """
    print("\n[Bayesian] Stage 1: Historical prior (Shanghai 2010-2025)...")
    prior = compute_historical_prior()
    top3_prior = prior.sort_values(ascending=False).head(3)
    for d, p in top3_prior.items():
        print(f"  {d:15s}: prior = {p:.4f}")

    print("[Bayesian] Stage 2: Sprint evidence update...")
    sprint_posterior = sprint_update(prior)
    top3_sprint = sprint_posterior.sort_values(ascending=False).head(3)
    for d, p in top3_sprint.items():
        print(f"  {d:15s}: sprint posterior = {p:.4f}")

    print("[Bayesian] Stage 3: GP qualifying grid update...")
    final_posterior = grid_update(sprint_posterior)

    print("[Bayesian] Top 5 final posterior:")
    for d, p in final_posterior.sort_values(ascending=False).head(5).items():
        print(f"  {d:15s}: {p:.4f}")

    print("[Bayesian] Computing 90% credible intervals...")
    ci_df = compute_credible_intervals(final_posterior)

    return final_posterior, ci_df


if __name__ == "__main__":
    posterior, ci = run_bayesian_model()
    print("\nFull Bayesian posterior + credible intervals:")
    merged = posterior.rename("bayes_prob").to_frame().join(ci[["ci_lo_90", "ci_hi_90"]])
    for driver, row in merged.sort_values("bayes_prob", ascending=False).iterrows():
        print(f"  {driver:15s}: {row['bayes_prob']:.4f}  [{row['ci_lo_90']:.4f}, {row['ci_hi_90']:.4f}]")
