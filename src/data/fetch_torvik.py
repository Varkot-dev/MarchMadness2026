"""
fetch_torvik.py — Download and parse Bart Torvik game-by-game results.

URL pattern: https://barttorvik.com/YYYY_results.csv
Data available: 2015–present (2013-2014 are empty on Torvik's server)

Output columns:
  YEAR, DATE, TEAM1, TEAM2, RESULT, WINNER, SCORE1, SCORE2,
  T1_EFF, T2_EFF, POSSESSIONS, T1_SCORE_RAW, T2_SCORE_RAW,
  T1_RANK, T2_RANK, [AVG_EFF, MARGIN] (2021+)
"""

import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from config import EXTERNAL_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TORVIK_DIR = EXTERNAL_DIR / "torvik"
BASE_URL = "https://barttorvik.com/{year}_results.csv"
REQUEST_DELAY = 1.5  # seconds between requests — be respectful

# Column schemas differ by year
COLS_10 = [
    "GAME_KEY", "DATE", "RESULT_STR",
    "T1_EFF", "T2_EFF", "POSSESSIONS",
    "T1_RAW", "T2_RAW", "T1_RANK", "T2_RANK",
]
COLS_11 = COLS_10 + ["AVG_EFF", "MARGIN"]

AVAILABLE_YEARS = list(range(2015, 2026))


def _fetch_raw(year: int) -> bytes:
    """Download raw CSV bytes for a given year."""
    url = BASE_URL.format(year=year)
    resp = requests.get(url, timeout=30, verify=False)
    resp.raise_for_status()
    if len(resp.content) == 0:
        raise ValueError(f"Empty response for year {year} — data not available")
    return resp.content


def _parse_game_key(game_key: str) -> tuple[str, str]:
    """
    Extract team1 and team2 from the concatenated GAME_KEY field.

    Torvik concatenates both team names + a date slug with no delimiter.
    Example: 'DukeNorth Carolina3-5' → ('Duke', 'North Carolina')

    Strategy: strip the trailing date pattern MM-DD or MM-DD-YY, then
    split remaining string on known team name boundaries using the DATE
    field for cross-reference — but since we don't have that here,
    we just return the raw key for downstream processing.

    Args:
        game_key: Raw concatenated string from column 0.

    Returns:
        Tuple (team1_raw, team2_raw) — best-effort extraction.
    """
    import re
    # Remove trailing date-like suffix: digits-digits or digits-digits-digits
    cleaned = re.sub(r"\d{1,2}-\d{1,2}(-\d{2,4})?$", "", game_key).strip()
    return cleaned, ""  # Team splitting requires DATE for disambiguation; done in parse_year


def parse_year(raw: bytes, year: int) -> pd.DataFrame:
    """
    Parse raw Torvik CSV bytes into a clean DataFrame for one year.

    Args:
        raw:  Raw CSV bytes from Torvik.
        year: Season year.

    Returns:
        DataFrame with one row per game.
    """
    import re

    # Detect actual column count from data rather than hardcoding by year
    probe = pd.read_csv(io.BytesIO(raw), header=None, nrows=5)
    n_cols = probe.shape[1]
    cols = COLS_11 if n_cols >= 11 else COLS_10

    df = pd.read_csv(io.BytesIO(raw), header=None, names=cols, on_bad_lines="skip")
    df["YEAR"] = year

    # Parse DATE
    df["DATE"] = pd.to_datetime(df["DATE"], format="%m/%d/%y", errors="coerce")

    # Extract winner and scores from RESULT_STR
    # Format: "TeamName -X.X, WS-LS (pct%)" or "TeamName (pct%)"
    result_pat = re.compile(
        r"^(?P<winner>.+?)\s*(?:[+-]\d+(?:\.\d+)?,\s*(?P<s1>\d+)-(?P<s2>\d+))?\s*\(\d+(?:\.\d+)?%\)"
    )

    def parse_result(s: str) -> pd.Series:
        if not isinstance(s, str):
            return pd.Series({"WINNER": None, "SCORE1": None, "SCORE2": None})
        m = result_pat.match(s.strip())
        if not m:
            return pd.Series({"WINNER": None, "SCORE1": None, "SCORE2": None})
        return pd.Series({
            "WINNER": m.group("winner").strip(),
            "SCORE1": int(m.group("s1")) if m.group("s1") else None,
            "SCORE2": int(m.group("s2")) if m.group("s2") else None,
        })

    result_parsed = df["RESULT_STR"].apply(parse_result)
    df = pd.concat([df, result_parsed], axis=1)

    # Extract teams from GAME_KEY using DATE as anchor
    # Format: Team1Name + Team2Name + MM-DD (as suffix of DATE)
    def extract_teams(row: pd.Series) -> pd.Series:
        key = str(row["GAME_KEY"])
        date = row["DATE"]
        if pd.isna(date):
            return pd.Series({"TEAM1": None, "TEAM2": None})
        suffix = date.strftime("%-m-%-d")
        if key.endswith(suffix):
            teams_str = key[: -len(suffix)].strip()
        else:
            teams_str = key
        # Winner is TEAM1 if listed first, but we just store both combined for now
        winner = row.get("WINNER", "")
        if isinstance(winner, str) and teams_str.startswith(winner):
            team1 = winner
            team2 = teams_str[len(winner):].strip()
        else:
            team1 = teams_str
            team2 = ""
        return pd.Series({"TEAM1": team1.strip(), "TEAM2": team2.strip()})

    teams = df.apply(extract_teams, axis=1)
    df = pd.concat([df, teams], axis=1)

    numeric_cols = ["T1_EFF", "T2_EFF", "POSSESSIONS", "T1_RAW", "T2_RAW", "T1_RANK", "T2_RANK"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[[
        "YEAR", "DATE", "GAME_KEY", "TEAM1", "TEAM2",
        "WINNER", "SCORE1", "SCORE2",
        "T1_EFF", "T2_EFF", "POSSESSIONS", "T1_RANK", "T2_RANK",
    ]]


def fetch_and_cache(year: int, force: bool = False) -> pd.DataFrame:
    """
    Fetch Torvik data for one year, using local cache if available.

    Args:
        year:  Season year (2015–2025).
        force: Re-download even if cached.

    Returns:
        Parsed DataFrame for the year.
    """
    TORVIK_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = TORVIK_DIR / f"torvik_{year}.csv"

    if cache_path.exists() and not force:
        log.info(f"{year}: loading from cache ({cache_path})")
        return pd.read_csv(cache_path, parse_dates=["DATE"])

    log.info(f"{year}: fetching from barttorvik.com ...")
    raw = _fetch_raw(year)
    df = parse_year(raw, year)
    df.to_csv(cache_path, index=False)
    log.info(f"{year}: {len(df)} games saved to {cache_path}")
    return df


def fetch_all(years: list[int] = AVAILABLE_YEARS, force: bool = False) -> pd.DataFrame:
    """
    Fetch and combine Torvik game data for all specified years.

    Args:
        years: List of season years to fetch.
        force: Re-download even if cached.

    Returns:
        Combined DataFrame sorted by DATE.
    """
    frames = []
    for year in years:
        try:
            df = fetch_and_cache(year, force=force)
            frames.append(df)
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            log.error(f"{year}: failed — {e}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["YEAR", "DATE"]).reset_index(drop=True)
    log.info(f"Total games fetched: {len(combined)} across years {years[0]}-{years[-1]}")
    return combined


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")  # suppress SSL verify=False warning

    df = fetch_all()
    print(f"\nShape: {df.shape}")
    print(f"Years: {sorted(df['YEAR'].unique())}")
    print(f"\nSample 2025:")
    print(df[df["YEAR"] == 2025].head(5).to_string(index=False))
