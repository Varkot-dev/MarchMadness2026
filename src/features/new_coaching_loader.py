"""
new_coaching_loader.py — Add COACH_PREMIUM to features_new.csv.

Uses two sources in priority order:
  1. Kaggle MTeamCoaches.csv (2008–2025) — exact team+year+coach mapping
  2. Manual 2026 overrides for known coaching changes

COACH_PREMIUM = PASE from Coach Results.csv (Performance Above Seed Expectation).
PASE = career tournament wins minus expected wins by seed, computed over all
prior seasons (no leakage — 2026 teams get PASE computed through 2025).

Coach Results.csv (in 2026 MarchMadness/ folder) contains pre-computed career
PASE for 332 coaches. We map team → coach → PASE.

Name fixes applied where Kaggle coach name format differs from Coach Results.

Output: features_new.csv updated with COACH and COACH_PREMIUM columns.

Usage:
    python3 -m src.features.new_coaching_loader
"""

import logging
from pathlib import Path

import pandas as pd

from config import EXTERNAL_DIR, PROCESSED_DIR
from src.features.coaching import CBB_TO_KAGGLE_NAMES

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

KAGGLE_COACHES_FILE = EXTERNAL_DIR / "kaggle" / "MTeamCoaches.csv"
KAGGLE_TEAMS_FILE   = EXTERNAL_DIR / "kaggle" / "MTeams.csv"
COACH_RESULTS_FILE  = Path("2026 MarchMadness") / "Coach Results.csv"

# Kaggle coach name (title-cased) → Coach Results COACH name
# Only entries where they differ
KAGGLE_TO_PASE_NAME: dict[str, str] = {
    "T J Otzelberger":   "TJ Otzelberger",
    "Mike White":        "Michael White",
    "Kenneth Blakeney":  "Kenny Blakeney",
    "Fran Mccaffery":    "Fran McCaffery",
    "Bill Courtney":     "",       # new coach, no PASE data
    "Ben Fletcher":      "",       # new coach, no PASE data
    "Jake Diebler":      "",       # new coach, no PASE data
    "Kyle Neptune":      "",       # new coach, no PASE data
    "Ron Sanchez":       "",       # new coach, no PASE data
    "Clint Sargent":     "",       # new coach, no PASE data
    "Speedy Claxton":    "",       # new coach, no PASE data
    "Josh Schertz":      "",       # new coach, no PASE data
    "Alex Pribble":      "",       # new coach, no PASE data
    "Rick Croy":         "",       # no PASE data
    "Antoine Pettway":   "",       # no PASE data
    "Grant Leonard":     "",       # no PASE data
    "Brian Collins":     "",       # no PASE data
    "Travis Steele":     "",       # no PASE data
    "Gerry Mcnamara":    "",       # new coach, no PASE data
    "Byron Smith":       "",       # no PASE data
}

# Manual coach assignments for teams missing from Kaggle 2025 data
MANUAL_2026_COACHES: dict[str, str] = {
    "LIU Brooklyn": "",   # 16-seed, PASE irrelevant
}


def _build_team_coach_map(year: int) -> dict[str, str]:
    """
    Build {new_dataset_team_name: coach_name} for a given year.

    Uses Kaggle MTeamCoaches for years 2008–2025.
    Applies MANUAL_2026_COACHES overrides for any missing entries.

    Args:
        year: Tournament year.

    Returns:
        Dict mapping team name (new dataset format) to coach name (title-cased).
    """
    kaggle_year = min(year, 2025)  # Kaggle only goes to 2025
    coaches_df = pd.read_csv(KAGGLE_COACHES_FILE)
    teams_df   = pd.read_csv(KAGGLE_TEAMS_FILE)

    yr_coaches = coaches_df[coaches_df["Season"] == kaggle_year].copy()

    # Keep only the coach with the most days coached (handles mid-season changes)
    yr_coaches["days"] = yr_coaches["LastDayNum"] - yr_coaches["FirstDayNum"]
    yr_coaches = (yr_coaches
                  .sort_values("days", ascending=False)
                  .drop_duplicates("TeamID", keep="first"))

    merged = yr_coaches.merge(teams_df[["TeamID", "TeamName"]], on="TeamID")
    merged["coach_clean"] = merged["CoachName"].str.replace("_", " ").str.title()

    # Invert CBB_TO_KAGGLE_NAMES: Kaggle team name → new dataset team name
    kaggle_to_new: dict[str, str] = {v: k for k, v in CBB_TO_KAGGLE_NAMES.items()}

    result: dict[str, str] = {}
    for _, row in merged.iterrows():
        kaggle_name = row["TeamName"]
        new_name = kaggle_to_new.get(kaggle_name, kaggle_name)
        result[new_name] = row["coach_clean"]

    result.update(MANUAL_2026_COACHES)
    return result


def _build_pase_lookup() -> dict[str, float]:
    """
    Build {coach_name: PASE} from Coach Results.csv.

    PASE = Performance Above Seed Expectation (career wins minus expected by seed).
    Pre-computed in the 2026 MarchMadness dataset.

    Returns:
        Dict mapping coach name (lower-cased) to PASE float.
    """
    if not COACH_RESULTS_FILE.exists():
        log.warning(f"Coach Results file not found: {COACH_RESULTS_FILE}")
        return {}

    cr = pd.read_csv(COACH_RESULTS_FILE)
    pase: dict[str, float] = {}
    for _, row in cr.iterrows():
        name = str(row["COACH"]).strip()
        pase[name.lower()] = float(row["PASE"]) if pd.notna(row["PASE"]) else 0.0

    log.info(f"Loaded PASE for {len(pase)} coaches from Coach Results.csv")
    return pase


def merge_coaching_into_features(
    features_df: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """
    Add COACH and COACH_PREMIUM columns to features_new DataFrame.

    COACH_PREMIUM = PASE value from Coach Results.csv for the team's coach.
    Teams/coaches with no PASE history default to 0.0 (neutral premium).

    Args:
        features_df: features_new DataFrame (YEAR + TEAM + feature columns).
        save:        If True, overwrite data/processed/features_new.csv.

    Returns:
        Updated DataFrame with COACH (str) and COACH_PREMIUM (float) columns.
    """
    pase_lookup = _build_pase_lookup()
    df = features_df.copy()

    coaches: list[str] = []
    premiums: list[float] = []

    for _, row in df.iterrows():
        year = int(row["YEAR"])
        team = row["TEAM"]

        coach_map = _build_team_coach_map(year)
        kaggle_coach = coach_map.get(team, "")

        # Apply name fixes for PASE lookup
        pase_name = KAGGLE_TO_PASE_NAME.get(kaggle_coach, kaggle_coach)

        if pase_name:
            pase = pase_lookup.get(pase_name.lower(), 0.0)
        else:
            pase = 0.0

        coaches.append(kaggle_coach)
        premiums.append(pase)

    df["COACH"] = coaches
    df["COACH_PREMIUM"] = premiums

    matched = sum(1 for p in premiums if p != 0.0)
    log.info(
        f"Coaching data: {matched}/{len(df)} rows have non-zero COACH_PREMIUM "
        f"({matched/len(df)*100:.1f}%)"
    )

    if save:
        out = PROCESSED_DIR / "features_new.csv"
        df.to_csv(out, index=False)
        log.info(f"Saved updated features_new.csv with COACH_PREMIUM → {out}")

    return df


def main() -> None:
    """Load features_new.csv, add coaching columns, save."""
    features_path = PROCESSED_DIR / "features_new.csv"
    if not features_path.exists():
        from src.features.new_data_loader import load_new_features
        df = load_new_features(save=False)
    else:
        df = pd.read_csv(features_path)

    df = merge_coaching_into_features(df, save=True)

    print(f"\n{'='*60}")
    print("2026 TOURNAMENT TEAMS — COACHING PREMIUM")
    print(f"{'='*60}")
    teams_2026 = df[df["YEAR"] == 2026].sort_values("COACH_PREMIUM", ascending=False)
    print(teams_2026[["TEAM", "SEED", "COACH", "COACH_PREMIUM"]].to_string(index=False))


if __name__ == "__main__":
    main()
