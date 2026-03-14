"""
Weather data and race condition modeling for 2026 China GP.
"""

import numpy as np

# ─── 2026 China GP Race Conditions ─────────────────────────────────────────────

RACE_CONDITIONS = {
    "air_temp_min":        16.0,
    "air_temp_max":        19.0,
    "track_temp":          33.0,
    "rain_prob_start":     0.07,   # 7% at lights out
    "rain_prob_peak":      0.25,   # up to 25% during race window
    "wind_speed_kmh":      11.0,   # ENE
    "wind_gust_kmh":       36.0,
    "humidity":            0.62,
    "cloud_cover":         0.55,
}

# Historical safety car rate at Shanghai
SHANGHAI_SC_RATE      = 0.467   # ~45% per race historically
SC_SPRINT_CONFIRMED   = True    # Sprint deployed SC → 1.2x multiplier
SC_RATE_ADJUSTED      = SHANGHAI_SC_RATE * 1.2  # = 0.56 (cap at 0.65)
SC_RATE_FINAL         = min(SC_RATE_ADJUSTED, 0.65)

# Per-lap safety car probability (56-lap race)
SC_PROB_PER_LAP_DRY   = SC_RATE_FINAL / 56   # ~0.0116 per lap
SC_PROB_PER_LAP_WET   = 0.80 / 56            # very high in wet

# ─── Wet Race Driver Advantage ─────────────────────────────────────────────────

# Historical wet-race win rate (relative boost vs dry)
# Based on 2008-2025 wet race analysis
WET_RACE_DRIVER_BOOST = {
    "hamilton":   1.45,   # Master of wet races (Brazil 2008, Germany 2018, etc.)
    "verstappen": 1.30,   # Also excellent in wet (Brazil 2016 as teenager, etc.)
    "alonso":     1.25,   # Legendary wet driver (Monaco 2004, 2007, etc.)
    "leclerc":    1.10,
    "norris":     1.08,
    "piastri":    1.00,
    "russell":    1.05,
    "antonelli":  0.90,   # Rookie, wet race uncertainty
    "gasly":      1.02,
    "sainz":      1.05,
    "bearman":    0.88,
    "hadjar":     0.85,
    "lawson":     0.92,
    "ocon":       0.95,
    "hulkenberg": 1.00,
    "bortoleto":  0.88,
    "albon":      0.98,
    "stroll":     0.92,
    "bottas":     0.95,
    "lindblad":   0.85,
    "doohan":     0.87,
    "crawford":   0.83,
}


def sample_rain_onset(n_sims: int, n_laps: int = 56) -> np.ndarray:
    """
    For each simulation, sample whether rain occurs and at which lap.
    Returns array of shape (n_sims,) where -1 = no rain, else = rain onset lap.
    """
    rain_onset = np.full(n_sims, -1, dtype=int)
    # 25% chance of rain at some point during the race
    rain_mask = np.random.random(n_sims) < RACE_CONDITIONS["rain_prob_peak"]
    n_rain = rain_mask.sum()
    if n_rain > 0:
        # Rain onset uniform across laps 5-45 (unlikely to rain at start/end)
        rain_laps = np.random.randint(5, 46, size=n_rain)
        rain_onset[rain_mask] = rain_laps
    return rain_onset


def get_weather_scenario(rain_onset_lap: int) -> dict:
    """Return a weather scenario dict for a simulation."""
    if rain_onset_lap < 0:
        return {
            "is_wet": False,
            "rain_lap": -1,
            "condition": "dry",
            "sc_prob_per_lap": SC_PROB_PER_LAP_DRY,
        }
    return {
        "is_wet": True,
        "rain_lap": rain_onset_lap,
        "condition": "wet",
        "sc_prob_per_lap": SC_PROB_PER_LAP_WET,
    }


if __name__ == "__main__":
    print("2026 China GP Race Conditions:")
    for k, v in RACE_CONDITIONS.items():
        print(f"  {k}: {v}")
    print(f"\nSafety Car Probability (adjusted): {SC_RATE_FINAL:.1%}/race")
    print(f"Per-lap SC probability (dry): {SC_PROB_PER_LAP_DRY:.4f}")

    rain = sample_rain_onset(1000)
    rain_count = (rain >= 0).sum()
    print(f"\nRain in {rain_count}/1000 simulations ({rain_count/10:.1f}%)")
