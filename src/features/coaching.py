"""
coaching.py — Compute Coaching Premium for each coach/year.

Formula: Coach_Premium = career_tournament_wins - expected_wins_by_seed

Uses an expanding window to avoid temporal leakage:
  When predicting year N, only use wins from years BEFORE N.

Coach → Team mapping comes from Kaggle's MTeamCoaches + MTeams files,
giving a direct YEAR + TeamID + CoachName lookup — no inference needed.

Output columns: COACH (str), COACH_PREMIUM (float)
"""

import logging
from pathlib import Path

import pandas as pd

from config import EXTERNAL_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

KAGGLE_COACHES_FILE = EXTERNAL_DIR / "kaggle" / "MTeamCoaches.csv"
KAGGLE_TEAMS_FILE = EXTERNAL_DIR / "kaggle" / "MTeams.csv"

# Historical average tournament wins by seed (1985–2024).
EXPECTED_WINS_BY_SEED: dict[int, float] = {
    1: 3.20, 2: 2.40, 3: 1.90, 4: 1.60,
    5: 1.10, 6: 1.00, 7: 0.90, 8: 0.70,
    9: 0.60, 10: 0.60, 11: 0.70, 12: 0.60,
    13: 0.30, 14: 0.20, 15: 0.10, 16: 0.02,
}

# Postseason result → tournament wins.
POSTSEASON_TO_WINS: dict[str, int] = {
    "Champions": 6, "2ND": 5, "F4": 4, "E8": 3,
    "S16": 2, "R32": 1, "R64": 0, "R68": 0,
}

# CBB dataset team names → Kaggle team names (only where they differ).
CBB_TO_KAGGLE_NAMES: dict[str, str] = {
    "Abilene Christian":    "Abilene Chr",
    "Alabama St.":          "Alabama St",
    "Albany":               "SUNY Albany",
    "Alcorn St.":           "Alcorn St",
    "American":             "American Univ",
    "Appalachian St.":      "Appalachian St",
    "Arizona St.":          "Arizona St",
    "Arkansas Little Rock": "Ark Little Rock",
    "Arkansas Pine Bluff":  "Ark Pine Bluff",
    "Arkansas St.":         "Arkansas St",
    "Ball St.":             "Ball St",
    "Bethune Cookman":      "Bethune-Cookman",
    "Boise St.":            "Boise St",
    "Boston University":    "Boston Univ",
    "Cal St. Bakersfield":  "CS Bakersfield",
    "Cal St. Fullerton":    "CS Fullerton",
    "Cal St. Northridge":   "CS Northridge",
    "Central Arkansas":     "Cent Arkansas",
    "Central Connecticut":  "Central Conn",
    "Central Michigan":     "C Michigan",
    "Charleston":           "Col Charleston",
    "Charlotte":            "UNC Charlotte",
    "Chicago St.":          "Chicago St",
    "Cleveland St.":        "Cleveland St",
    "Coastal Carolina":     "Coastal Car",
    "Colorado St.":         "Colorado St",
    "Dixie St.":            "Dixie St",
    "East Tennessee St.":   "ETSU",
    "Eastern Illinois":     "E Illinois",
    "Eastern Kentucky":     "E Kentucky",
    "Eastern Michigan":     "E Michigan",
    "Eastern Washington":   "E Washington",
    "FIU":                  "Florida Intl",
    "Florida Atlantic":     "FL Atlantic",
    "Fairleigh Dickinson":  "F Dickinson",
    "Florida Gulf Coast":   "FGCU",
    "Fort Wayne":           "IPFW",
    "Fresno St.":           "Fresno St",
    "Gardner Webb":         "Gardner-Webb",
    "Georgia St.":          "Georgia St",
    "Grambling St.":        "Grambling",
    "Houston Baptist":      "Houston Chr",
    "Idaho St.":            "Idaho St",
    "IU Indy":              "IUPUI",
    "Illinois Chicago":     "IL Chicago",
    "Iowa St.":             "Iowa St",
    "Kansas St.":           "Kansas St",
    "Kent St.":             "Kent",
    "LIU":                  "LIU Brooklyn",
    "Long Beach St.":       "Long Beach St",
    "Louisiana Lafayette":  "Louisiana",
    "Louisiana Monroe":     "ULM",
    "Loyola Chicago":       "Loyola-Chicago",
    "Maryland Eastern Shore": "MD E Shore",
    "McNeese St.":          "McNeese St",
    "Michigan St.":         "Michigan St",
    "Middle Tennessee":     "Middle Tenn",
    "Mississippi St.":      "Mississippi St",
    "Missouri St.":         "Missouri St",
    "Montana St.":          "Montana St",
    "Morehead St.":         "Morehead St",
    "Murray St.":           "Murray St",
    "New Mexico St.":       "New Mexico St",
    "Nicholls St.":         "Nicholls St",
    "North Carolina A&T":   "NC A&T",
    "North Carolina St.":   "NC State",
    "North Dakota St.":     "N Dakota St",
    "Northern Arizona":     "N Arizona",
    "Northern Colorado":    "N Colorado",
    "Northern Illinois":    "N Illinois",
    "Northern Iowa":        "N Iowa",
    "Northwestern St.":     "Northwestern St",
    "Ohio St.":             "Ohio St",
    "Oklahoma St.":         "Oklahoma St",
    "Oregon St.":           "Oregon St",
    "Penn St.":             "Penn St",
    "Portland St.":         "Portland St",
    "Purdue Fort Wayne":    "Purdue FW",
    "Sacramento St.":       "Sacramento St",
    "Saint Mary's":         "St Mary's CA",
    "Saint Peter's":        "St Peter's",
    "Sam Houston St.":      "Sam Houston St",
    "San Diego St.":        "San Diego St",
    "SIU Edwardsville":     "SIUE",
    "Southeastern Louisiana": "SE Louisiana",
    "South Dakota St.":     "S Dakota St",
    "Southern Illinois":    "S Illinois",
    "Southern Utah":        "S Utah",
    "Stephen F. Austin":    "SF Austin",
    "Texas A&M Commerce":   "TX A&M Commerce",
    "Texas A&M Corpus Chris": "TAM C. Christi",
    "Texas St.":            "TX St San Marcos",
    "Southwest Texas St.":  "TX St San Marcos",
    "UT Rio Grande Valley": "UTRGV",
    "UTSA":                 "UT San Antonio",
    "Washington St.":       "Washington St",
    "Utah St.":             "Utah St",
    "Weber St.":            "Weber St",
    "Western Illinois":     "W Illinois",
    "Western Kentucky":     "WKU",
    "Western Michigan":     "W Michigan",
    "Wichita St.":          "Wichita St",
    "Wright St.":           "Wright St",
    # Additional name mappings for tournament experience feature coverage
    "Akron":                "Akron",
    "Bryant":               "Bryant",
    "College of Charleston": "Col Charleston",
    "Delaware":             "Delaware",
    "Florida St.":          "Florida St",
    "Gardner Webb":         "Gardner Webb",
    "George Washington":    "G Washington",
    "Grand Canyon":         "Grand Canyon",
    "Green Bay":            "WI Green Bay",
    "High Point":           "High Point",
    "Howard":               "Howard",
    "Jacksonville St.":     "Jacksonville St",
    "Kennesaw St.":         "Kennesaw",
    "Liberty":              "Liberty",
    "Lipscomb":             "Lipscomb",
    "Loyola Chicago":       "Loyola-Chicago",
    "Milwaukee":            "WI Milwaukee",
    "Montana":              "Montana",
    "Mount St. Mary's":     "Mt St Mary's",
    "Nebraska Omaha":       "NE Omaha",
    "Norfolk St.":          "Norfolk St",
    "North Carolina Central": "NC Central",
    "Northern Iowa":        "N Iowa",
    "Northern Kentucky":    "N Kentucky",
    "Northwestern St.":     "Northwestern St",
    "Prairie View A&M":     "Prairie View",
    "Robert Morris":        "Robert Morris",
    "Saint Francis":        "St Francis PA",
    "Saint Joseph's":       "St Joseph's PA",
    "Saint Louis":          "St Louis",
    "SIU Edwardsville":     "SIUE",
    "Southeast Missouri St.": "SE Missouri St",
    "Southern":             "Southern Univ",
    "St. Bonaventure":      "St Bonaventure",
    "St. John's":           "St John's",
    "Texas Southern":       "TX Southern",
    "Troy":                 "Troy",
    "UC San Diego":         "UC San Diego",
    "UNC Wilmington":       "UNC Wilmington",
    "Vermont":              "Vermont",
    "Wofford":              "Wofford",
    "Wyoming":              "Wyoming",
}


# ── Pure functions ─────────────────────────────────────────────────────────────

def expected_wins(seed: float) -> float:
    """
    Return historical average tournament wins for a given seed.

    Args:
        seed: Tournament seed (1–16).

    Returns:
        Expected wins, or 0.5 for unknown seeds.
    """
    if pd.isna(seed):
        return 0.5
    return EXPECTED_WINS_BY_SEED.get(int(seed), 0.5)


def load_kaggle_coach_lookup(
    coaches_path: Path = KAGGLE_COACHES_FILE,
    teams_path: Path = KAGGLE_TEAMS_FILE,
) -> pd.DataFrame:
    """
    Build a direct YEAR + KAGGLE_TEAM + COACH lookup from Kaggle files.

    Handles mid-season coach changes by keeping the coach with the most
    days coached (LastDayNum - FirstDayNum).

    Args:
        coaches_path: Path to MTeamCoaches.csv.
        teams_path:   Path to MTeams.csv.

    Returns:
        DataFrame with columns: YEAR (int), KAGGLE_TEAM (str), COACH (str).
    """
    coaches = pd.read_csv(coaches_path)
    teams = pd.read_csv(teams_path)

    df = coaches.merge(teams[["TeamID", "TeamName"]], on="TeamID")
    df["COACH"] = df["CoachName"].str.replace("_", " ").str.title()
    df["DAYS"] = df["LastDayNum"] - df["FirstDayNum"]
    df = (
        df.sort_values("DAYS", ascending=False)
        .drop_duplicates(subset=["Season", "TeamName"], keep="first")
        .rename(columns={"Season": "YEAR", "TeamName": "KAGGLE_TEAM"})
        [["YEAR", "KAGGLE_TEAM", "COACH"]]
        .reset_index(drop=True)
    )
    log.info(f"Kaggle lookup: {len(df)} team-year-coach records")
    return df


def compute_coach_premiums(team_coach_seed: pd.DataFrame) -> pd.DataFrame:
    """
    Compute expanding-window coaching premium for each coach/year.

    Premium = sum(actual wins before year N) - sum(expected wins before year N).
    First appearance gets premium = 0.0 (no prior history).

    Args:
        team_coach_seed: DataFrame with columns YEAR, COACH, TOURNEY_W, SEED.
                         One row per tournament appearance.

    Returns:
        DataFrame with columns: YEAR, COACH, COACH_PREMIUM.
    """
    result_rows = []

    for coach, group in team_coach_seed.groupby("COACH"):
        group = group.sort_values("YEAR")
        prior_wins = 0.0
        prior_expected = 0.0

        for _, row in group.iterrows():
            result_rows.append({
                "YEAR": row["YEAR"],
                "COACH": coach,
                "COACH_PREMIUM": round(prior_wins - prior_expected, 3),
            })
            # Accumulate AFTER recording — expanding window, no leakage
            prior_wins += row["TOURNEY_W"]
            prior_expected += expected_wins(row["SEED"])

    return pd.DataFrame(result_rows)


# ── Pipeline helpers ───────────────────────────────────────────────────────────

def _attach_coaches(df: pd.DataFrame, kaggle_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Join COACH column onto df via Kaggle lookup using YEAR + mapped team name.

    Args:
        df:             CBB features DataFrame with YEAR and TEAM columns.
        kaggle_lookup:  Output of load_kaggle_coach_lookup().

    Returns:
        df with COACH column added (NaN where no match found).
    """
    kaggle_teams = kaggle_lookup["KAGGLE_TEAM"].unique().tolist()
    name_map = {
        team: (team if team in set(kaggle_teams) else CBB_TO_KAGGLE_NAMES.get(team))
        for team in df["TEAM"].unique()
    }
    df = df.copy()
    df["KAGGLE_TEAM"] = df["TEAM"].map(name_map)
    df = df.merge(kaggle_lookup, on=["YEAR", "KAGGLE_TEAM"], how="left")
    df = df.drop(columns=["KAGGLE_TEAM"])

    matched = df["COACH"].notna().sum()
    log.info(f"Coach matched: {matched}/{len(df)} rows ({matched / len(df) * 100:.1f}%)")
    return df


def _build_premiums(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute coach premiums from tournament rows in df.

    Args:
        df: Features DataFrame with YEAR, COACH, SEED, POSTSEASON columns.

    Returns:
        DataFrame with YEAR, COACH, COACH_PREMIUM.
    """
    tourney = df[df["SEED"].notna() & df["COACH"].notna() & (df["YEAR"] != 2020)].copy()
    tourney["TOURNEY_W"] = tourney["POSTSEASON"].map(POSTSEASON_TO_WINS).fillna(0)
    return compute_coach_premiums(tourney[["YEAR", "COACH", "TOURNEY_W", "SEED"]])


# ── Public API ─────────────────────────────────────────────────────────────────

def build_coaching_features(
    momentum_path: Path = PROCESSED_DIR / "features_momentum.csv",
    coaches_path: Path = KAGGLE_COACHES_FILE,
    teams_path: Path = KAGGLE_TEAMS_FILE,
) -> pd.DataFrame:
    """
    Add COACH and COACH_PREMIUM columns to the features DataFrame.

    Args:
        momentum_path: Path to momentum features CSV (base DataFrame).
        coaches_path:  Path to Kaggle MTeamCoaches.csv.
        teams_path:    Path to Kaggle MTeams.csv.

    Returns:
        DataFrame with COACH (str) and COACH_PREMIUM (float) columns added.
    """
    df = pd.read_csv(momentum_path)
    kaggle_lookup = load_kaggle_coach_lookup(coaches_path, teams_path)

    df = _attach_coaches(df, kaggle_lookup)
    premiums = _build_premiums(df)

    df = df.merge(premiums[["YEAR", "COACH", "COACH_PREMIUM"]], on=["YEAR", "COACH"], how="left")
    df["COACH_PREMIUM"] = df["COACH_PREMIUM"].fillna(0.0)

    tourney = df[df["SEED"].notna() & (df["YEAR"] != 2020) & (df["YEAR"] != 2025)]
    log.info(
        f"COACH_PREMIUM range (tourney): "
        f"{tourney['COACH_PREMIUM'].min():.2f} to {tourney['COACH_PREMIUM'].max():.2f}"
    )
    return df


def save(df: pd.DataFrame, filename: str = "features_coaching.csv") -> Path:
    """Save coaching features to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / filename
    df.to_csv(out, index=False)
    log.info(f"Saved to {out}")
    return out


if __name__ == "__main__":
    df = build_coaching_features()
    save(df)

    tourney = df[
        df["SEED"].notna() &
        (df["YEAR"] != 2020) &
        (df["YEAR"] != 2025) &
        df["COACH"].notna()
    ]

    print("\nTop 10 Coaching Premium (tournament teams):")
    print(
        tourney.nlargest(10, "COACH_PREMIUM")[
            ["YEAR", "TEAM", "COACH", "SEED", "COACH_PREMIUM"]
        ].to_string(index=False)
    )

    print("\nTom Izzo career premium:")
    print(
        tourney[tourney["COACH"] == "Tom Izzo"][
            ["YEAR", "TEAM", "SEED", "COACH_PREMIUM"]
        ].to_string(index=False)
    )

    dups = df[df["COACH"].notna()].groupby(["YEAR", "COACH"])["TEAM"].nunique()
    bad = dups[dups > 1]
    print(f"\nCoach-year duplicates (should be 0): {len(bad)}")
