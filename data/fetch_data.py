"""
Data ingestion from OpenF1, Jolpica-F1, and FastF1.
All data is cached locally to avoid repeated API calls.
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
OPENF1_BASE  = "https://api.openf1.org/v1"


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _load_cache(key: str):
    p = _cache_path(key)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _save_cache(key: str, data):
    with open(_cache_path(key), "w") as f:
        json.dump(data, f)


def _get_json(url: str, params: dict = None, cache_key: str = None, retries: int = 3):
    if cache_key:
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if cache_key:
                _save_cache(cache_key, data)
            return data
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [WARN] Failed to fetch {url}: {e}")
                return None
            time.sleep(2 ** attempt)
    return None


# ─── Jolpica-F1 Historical Results ─────────────────────────────────────────────

def fetch_season_results(year: int) -> pd.DataFrame:
    """Fetch all race results for a given season."""
    key = f"jolpica_results_{year}"
    data = _get_json(
        f"{JOLPICA_BASE}/{year}/results.json",
        params={"limit": 1000},
        cache_key=key
    )
    if not data:
        return pd.DataFrame()
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    rows = []
    for race in races:
        circuit = race.get("Circuit", {}).get("circuitId", "")
        race_name = race.get("raceName", "")
        round_num = int(race.get("round", 0))
        date = race.get("date", "")
        for res in race.get("Results", []):
            rows.append({
                "year": year,
                "round": round_num,
                "race_name": race_name,
                "circuit_id": circuit,
                "date": date,
                "driver_id": res["Driver"]["driverId"],
                "driver_code": res["Driver"].get("code", ""),
                "constructor_id": res["Constructor"]["constructorId"],
                "grid": int(res.get("grid", 0)),
                "position": int(res["position"]) if res.get("position", "").isdigit() else 99,
                "status": res.get("status", ""),
                "points": float(res.get("points", 0)),
                "laps": int(res.get("laps", 0)),
                "is_winner": 1 if res.get("position") == "1" else 0,
            })
    return pd.DataFrame(rows)


def fetch_historical_results(start_year: int = 2010, end_year: int = 2025) -> pd.DataFrame:
    """Load all race results from start_year to end_year."""
    key = f"historical_results_{start_year}_{end_year}"
    cache_file = CACHE_DIR / f"{key}.parquet"
    if cache_file.exists():
        print(f"  [CACHE] Loading historical results from {cache_file}")
        return pd.read_parquet(cache_file)
    print(f"  Fetching historical race results {start_year}-{end_year}...")
    dfs = []
    for yr in range(start_year, end_year + 1):
        print(f"    Season {yr}...", end=" ", flush=True)
        df = fetch_season_results(yr)
        if not df.empty:
            dfs.append(df)
            print(f"{len(df)} rows")
        else:
            print("no data")
        time.sleep(0.3)
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet(cache_file)
    print(f"  Saved {len(combined)} rows to {cache_file}")
    return combined


def fetch_qualifying_results(year: int) -> pd.DataFrame:
    """Fetch qualifying results for a full season."""
    key = f"jolpica_quali_{year}"
    data = _get_json(
        f"{JOLPICA_BASE}/{year}/qualifying.json",
        params={"limit": 1000},
        cache_key=key
    )
    if not data:
        return pd.DataFrame()
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    rows = []
    for race in races:
        circuit = race.get("Circuit", {}).get("circuitId", "")
        round_num = int(race.get("round", 0))
        for res in race.get("QualifyingResults", []):
            rows.append({
                "year": year,
                "round": round_num,
                "circuit_id": circuit,
                "driver_id": res["Driver"]["driverId"],
                "constructor_id": res["Constructor"]["constructorId"],
                "quali_position": int(res.get("position", 99)),
                "q1": res.get("Q1", ""),
                "q2": res.get("Q2", ""),
                "q3": res.get("Q3", ""),
            })
    return pd.DataFrame(rows)


def fetch_sprint_results(year: int, round_num: int) -> pd.DataFrame:
    """Fetch sprint race results for a specific round."""
    key = f"jolpica_sprint_{year}_{round_num}"
    data = _get_json(
        f"{JOLPICA_BASE}/{year}/{round_num}/sprint.json",
        cache_key=key
    )
    if not data:
        return pd.DataFrame()
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    rows = []
    for race in races:
        for res in race.get("SprintResults", []):
            rows.append({
                "year": year,
                "round": round_num,
                "driver_id": res["Driver"]["driverId"],
                "constructor_id": res["Constructor"]["constructorId"],
                "sprint_grid": int(res.get("grid", 0)),
                "sprint_position": int(res["position"]) if res.get("position", "").isdigit() else 99,
                "sprint_status": res.get("status", ""),
                "sprint_points": float(res.get("points", 0)),
            })
    return pd.DataFrame(rows)


def fetch_all_sprint_history(start_year: int = 2021, end_year: int = 2025) -> pd.DataFrame:
    """Fetch all sprint race results to learn sprint-to-GP correlation."""
    # Sprint weekends by year/round
    sprint_rounds = {
        2021: [10, 15, 19],
        2022: [4, 11, 21],
        2023: [4, 8, 13, 17, 19, 21],
        2024: [3, 6, 11, 15, 19, 21],
        2025: [3, 6, 11, 15],
    }
    key = "all_sprint_history"
    cache_file = CACHE_DIR / f"{key}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    dfs = []
    for yr, rounds in sprint_rounds.items():
        if yr < start_year or yr > end_year:
            continue
        for rnd in rounds:
            df = fetch_sprint_results(yr, rnd)
            if not df.empty:
                dfs.append(df)
            time.sleep(0.3)
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet(cache_file)
    return combined


# ─── OpenF1 Session Data ────────────────────────────────────────────────────────

def fetch_openf1_sessions(year: int, country: str = "China") -> list:
    """Fetch session info for a specific GP weekend."""
    key = f"openf1_sessions_{year}_{country.lower()}"
    data = _get_json(
        f"{OPENF1_BASE}/sessions",
        params={"year": year, "country_name": country},
        cache_key=key
    )
    return data or []


def fetch_openf1_laps(session_key: int, driver_number: int = None) -> list:
    """Fetch lap data for a session (optionally filter by driver)."""
    params = {"session_key": session_key}
    if driver_number:
        params["driver_number"] = driver_number
    key = f"openf1_laps_{session_key}_{driver_number or 'all'}"
    data = _get_json(f"{OPENF1_BASE}/laps", params=params, cache_key=key)
    return data or []


def fetch_openf1_drivers(session_key: int) -> list:
    """Fetch driver info for a session."""
    key = f"openf1_drivers_{session_key}"
    data = _get_json(
        f"{OPENF1_BASE}/drivers",
        params={"session_key": session_key},
        cache_key=key
    )
    return data or []


def fetch_openf1_weather(session_key: int) -> list:
    """Fetch weather data for a session."""
    key = f"openf1_weather_{session_key}"
    data = _get_json(
        f"{OPENF1_BASE}/weather",
        params={"session_key": session_key},
        cache_key=key
    )
    return data or []


# ─── Constructor Standings ──────────────────────────────────────────────────────

def fetch_constructor_standings(year: int, round_num: int = None) -> pd.DataFrame:
    """Fetch constructor standings after a specific round."""
    url = f"{JOLPICA_BASE}/{year}"
    if round_num:
        url += f"/{round_num}"
    url += "/constructorStandings.json"
    key = f"constructor_standings_{year}_{round_num or 'final'}"
    data = _get_json(url, cache_key=key)
    if not data:
        return pd.DataFrame()
    lists = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    rows = []
    for sl in lists:
        for cs in sl.get("ConstructorStandings", []):
            rows.append({
                "year": year,
                "constructor_id": cs["Constructor"]["constructorId"],
                "points": float(cs.get("points", 0)),
                "position": int(cs.get("position", 0)),
                "wins": int(cs.get("wins", 0)),
            })
    return pd.DataFrame(rows)


# ─── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("F1 Data Ingestion — 2026 China GP V2")
    print("=" * 60)

    print("\n[1/4] Fetching historical race results (2010–2025)...")
    hist = fetch_historical_results(2010, 2025)
    print(f"  Total rows: {len(hist)}")

    print("\n[2/4] Fetching sprint race history (2021–2025)...")
    sprints = fetch_all_sprint_history()
    print(f"  Sprint rows: {len(sprints)}")

    print("\n[3/4] Fetching 2025 constructor standings...")
    standings = fetch_constructor_standings(2025)
    print(f"  Constructors: {len(standings)}")

    print("\n[4/4] Probing OpenF1 for 2026 China sessions...")
    sessions = fetch_openf1_sessions(2026, "China")
    print(f"  Sessions found: {len(sessions)}")
    for s in sessions:
        print(f"    {s.get('session_type','?')} — key={s.get('session_key','?')}")

    print("\nData ingestion complete.")
