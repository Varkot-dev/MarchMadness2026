"""
tourney_experience.py — Compute prior NCAA tournament experience features.

For each team in year Y, sum all tournament minutes played across seasons
strictly before Y (using MNCAATourneyDetailedResults.csv). This captures
institutional program experience — teams like Kansas/Duke who go deep every
year accumulate far more minutes than first-time qualifiers.

Features produced:
  - TOURNEY_EXP_MINUTES : raw cumulative minutes in prior NCAA tournaments
  - TOURNEY_EXP_LOG     : log1p(TOURNEY_EXP_MINUTES) — dampens diminishing returns

Leakage note: strictly uses seasons < Y, so no future data leaks in.
Roster caveat: measures program experience, not individual player experience.
               Validated via correlation with rounds_won before use in model.

Usage:
    from src.features.tourney_experience import build_tourney_experience
    df = build_tourney_experience()   # returns DataFrame keyed on (TeamID, Season)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import EXTERNAL_DIR, PROCESSED_DIR

log = logging.getLogger(__name__)


def build_tourney_experience(
    results_path: Path = EXTERNAL_DIR / "kaggle" / "MNCAATourneyDetailedResults.csv",
    seeds_path:   Path = EXTERNAL_DIR / "kaggle" / "MNCAATourneySeeds.csv",
    teams_path:   Path = EXTERNAL_DIR / "kaggle" / "MTeams.csv",
    min_season: int = 2013,
    max_season: int = 2025,
) -> pd.DataFrame:
    """
    Build prior tournament experience features for all tournament teams 2013-2025.

    For each (TeamID, Season=Y), sums minutes from all tournament games
    in seasons strictly < Y. Minutes per game = 40 + (NumOT * 5).

    Args:
        results_path: Path to MNCAATourneyDetailedResults.csv
        seeds_path:   Path to MNCAATourneySeeds.csv (to get tournament team list)
        teams_path:   Path to MTeams.csv (for logging team names)
        min_season:   First season to compute features for (default 2013)
        max_season:   Last season to compute features for (default 2025)

    Returns:
        DataFrame with columns [TeamID, TeamName, Season,
                                 TOURNEY_EXP_MINUTES, TOURNEY_EXP_LOG]
        One row per (team, season) that appeared in the tournament.
    """
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    if not seeds_path.exists():
        raise FileNotFoundError(f"Seeds file not found: {seeds_path}")

    results = pd.read_csv(results_path)
    seeds   = pd.read_csv(seeds_path)
    teams   = pd.read_csv(teams_path)

    log.info(f"Loaded {len(results)} tournament games ({results.Season.min()}–{results.Season.max()})")

    # Compute minutes per game for every game
    results = results.copy()
    results["minutes"] = 40 + results["NumOT"] * 5

    # Build long-form: one row per (team, season, minutes) for winners and losers
    winners = results[["Season", "WTeamID", "minutes"]].rename(columns={"WTeamID": "TeamID"})
    losers  = results[["Season", "LTeamID", "minutes"]].rename(columns={"LTeamID": "TeamID"})
    all_games = pd.concat([winners, losers], ignore_index=True)

    # For each tournament team in target seasons, sum minutes from prior seasons only
    target_teams = seeds[seeds["Season"].between(min_season, max_season)][["Season", "TeamID"]].drop_duplicates()

    rows = []
    for _, row in target_teams.iterrows():
        season  = int(row["Season"])
        team_id = int(row["TeamID"])

        prior = all_games[(all_games["TeamID"] == team_id) & (all_games["Season"] < season)]
        exp_minutes = int(prior["minutes"].sum())

        rows.append({
            "TeamID":              team_id,
            "Season":              season,
            "TOURNEY_EXP_MINUTES": exp_minutes,
            "TOURNEY_EXP_LOG":     float(np.log1p(exp_minutes)),
        })

    df = pd.DataFrame(rows)

    # Attach team names for readability
    df = df.merge(teams[["TeamID", "TeamName"]], on="TeamID", how="left")
    df = df[["TeamID", "TeamName", "Season", "TOURNEY_EXP_MINUTES", "TOURNEY_EXP_LOG"]]
    df = df.sort_values(["Season", "TOURNEY_EXP_MINUTES"], ascending=[True, False]).reset_index(drop=True)

    log.info(f"Built experience features for {len(df)} team-seasons ({min_season}–{max_season})")
    log.info(f"  Min exp: {df.TOURNEY_EXP_MINUTES.min()} min  |  Max: {df.TOURNEY_EXP_MINUTES.max()} min")
    log.info(f"  Teams with zero prior experience: {(df.TOURNEY_EXP_MINUTES == 0).sum()}")

    return df


def validate_experience_feature(df: pd.DataFrame, results_path: Path = EXTERNAL_DIR / "kaggle" / "MNCAATourneyDetailedResults.csv") -> pd.DataFrame:
    """
    Validate that TOURNEY_EXP_MINUTES correlates with tournament performance.

    Merges experience with actual rounds won per team-season and computes
    Pearson correlation. Prints a summary table by experience quartile.

    Args:
        df:           Output of build_tourney_experience()
        results_path: Path to MNCAATourneyDetailedResults.csv

    Returns:
        DataFrame with columns [TeamID, Season, TOURNEY_EXP_MINUTES,
                                 TOURNEY_EXP_LOG, rounds_won]
    """
    from src.evaluation.backtest import _build_daynum_to_round

    results = pd.read_csv(results_path)

    # Compute rounds_won per team per season from actual results
    rounds_won = {}
    for season, grp in results.groupby("Season"):
        d2r = _build_daynum_to_round(grp)
        for _, row in grp.iterrows():
            rnd = d2r.get(int(row["DayNum"]))
            if rnd is None:
                continue
            w, l = int(row["WTeamID"]), int(row["LTeamID"])
            rounds_won[(season, w)] = max(rounds_won.get((season, w), 0), rnd)
            rounds_won[(season, l)] = max(rounds_won.get((season, l), 0), rnd - 1)

    df = df.copy()
    df["rounds_won"] = df.apply(lambda r: rounds_won.get((r["Season"], r["TeamID"]), 0), axis=1)

    corr_raw = df["TOURNEY_EXP_MINUTES"].corr(df["rounds_won"])
    corr_log = df["TOURNEY_EXP_LOG"].corr(df["rounds_won"])
    log.info(f"Correlation with rounds_won — raw: {corr_raw:.3f}  |  log: {corr_log:.3f}")

    # Quartile breakdown
    df["exp_quartile"] = pd.qcut(df["TOURNEY_EXP_MINUTES"], q=4, labels=["Q1 (least)", "Q2", "Q3", "Q4 (most)"])
    summary = df.groupby("exp_quartile", observed=True)["rounds_won"].agg(["mean", "count"]).round(3)
    log.info(f"\nRounds won by experience quartile:\n{summary.to_string()}")

    return df


def merge_into_features(
    features_path: Path = PROCESSED_DIR / "features_coaching.csv",
) -> pd.DataFrame:
    """
    Merge TOURNEY_EXP_MINUTES and TOURNEY_EXP_LOG into features_coaching.csv.

    Joins on (TEAM, YEAR) using the CBB_TO_KAGGLE_NAMES map from coaching.py
    to bridge name formats. Teams with no match get NaN (not zero — model
    should treat missing and zero differently via imputation).

    Args:
        features_path: Path to features_coaching.csv

    Returns:
        Updated DataFrame with two new columns added.
        Also overwrites features_coaching.csv in place.
    """
    from src.features.coaching import CBB_TO_KAGGLE_NAMES

    feats = pd.read_csv(features_path)
    teams = pd.read_csv(EXTERNAL_DIR / "kaggle" / "MTeams.csv")

    # Build cbb_name -> kaggle TeamID using CBB_TO_KAGGLE_NAMES as the bridge
    # kaggle TeamName -> TeamID
    kaggle_name_to_id = teams.set_index("TeamName")["TeamID"].to_dict()

    # CBB name -> TeamID (direct match first, then via CBB_TO_KAGGLE_NAMES)
    cbb_to_id: dict[str, int] = {}
    for cbb_name in feats["TEAM"].unique():
        kaggle_name = CBB_TO_KAGGLE_NAMES.get(cbb_name, cbb_name)
        tid = kaggle_name_to_id.get(kaggle_name)
        if tid is not None:
            cbb_to_id[cbb_name] = int(tid)

    log.info(f"CBB->Kaggle ID mapping: {len(cbb_to_id)}/{feats['TEAM'].nunique()} teams resolved")

    exp = build_tourney_experience()
    exp_lookup = exp.set_index(["TeamID", "Season"])[["TOURNEY_EXP_MINUTES", "TOURNEY_EXP_LOG"]]

    def get_exp(row) -> pd.Series:
        tid = cbb_to_id.get(row["TEAM"])
        yr  = int(row["YEAR"]) if pd.notna(row.get("YEAR")) else None
        if tid is None or yr is None:
            return pd.Series({"TOURNEY_EXP_MINUTES": np.nan, "TOURNEY_EXP_LOG": np.nan})
        key = (tid, yr)
        return exp_lookup.loc[key] if key in exp_lookup.index else pd.Series({"TOURNEY_EXP_MINUTES": np.nan, "TOURNEY_EXP_LOG": np.nan})

    exp_cols = feats.apply(get_exp, axis=1)
    feats["TOURNEY_EXP_MINUTES"] = exp_cols["TOURNEY_EXP_MINUTES"]
    feats["TOURNEY_EXP_LOG"]     = exp_cols["TOURNEY_EXP_LOG"]

    tourney = feats["SEED"].notna()
    filled  = feats.loc[tourney, "TOURNEY_EXP_MINUTES"].notna().sum()
    total   = tourney.sum()
    log.info(f"Tournament rows: {total} | With experience data: {filled} ({filled/total*100:.1f}%)")

    missing_teams = feats.loc[tourney & feats["TOURNEY_EXP_MINUTES"].isna(), "TEAM"].unique()
    if len(missing_teams):
        log.warning(f"Still missing {len(missing_teams)} tournament teams: {list(missing_teams)}")

    feats.to_csv(features_path, index=False)
    log.info(f"Saved updated features to {features_path}")
    return feats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = build_tourney_experience()

    print("\n=== Top 10 most experienced teams (2024) ===")
    print(df[df.Season == 2024].head(10)[["TeamName", "TOURNEY_EXP_MINUTES", "TOURNEY_EXP_LOG"]].to_string(index=False))

    print("\n=== Validation: does experience predict rounds won? ===")
    validate_experience_feature(df)

    print("\n=== Merging into features_coaching.csv ===")
    merge_into_features()
