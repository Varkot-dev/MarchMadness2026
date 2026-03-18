"""
new_matchup_builder.py — Build binary matchup training data from the new dataset.

Uses Kaggle MNCAATourneyDetailedResults.csv for explicit winner/loser game pairs
(reliable), then looks up features from features_new.csv (the new enriched dataset).

This approach is correct because:
- Kaggle has explicit WTeamID/LTeamID per game — no heuristic opponent pairing needed
- New dataset has clean features for 2008–2026
- Name bridging: Kaggle ID → CBB name via CBB_TO_KAGGLE_NAMES → new dataset TEAM name

For each game (A beats B):
    row1: features(A) - features(B), LABEL=1
    row2: features(B) - features(A), LABEL=0

Output saved to data/processed/matchups_new.csv

Usage:
    from src.features.new_matchup_builder import build_matchup_data
    from src.features.new_data_loader import load_new_features

    features = load_new_features()
    matchups = build_matchup_data(features)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import EXTERNAL_DIR, PROCESSED_DIR

log = logging.getLogger(__name__)

KAGGLE_DIR     = EXTERNAL_DIR / "kaggle"
RESULTS_FILE   = KAGGLE_DIR / "MNCAATourneyDetailedResults.csv"
TEAMS_FILE     = KAGGLE_DIR / "MTeams.csv"


def _build_kaggle_to_new_name(
    features_df: pd.DataFrame,
    teams_df: pd.DataFrame,
) -> dict[int, str]:
    """
    Map Kaggle TeamID → new-dataset TEAM name string.

    CBB_TO_KAGGLE_NAMES maps new-dataset names → Kaggle names (e.g.
    "Arizona St." → "Arizona State"). We invert it to get
    Kaggle name → new-dataset name, then look up by Kaggle TeamID.

    Strategy:
      1. CBB_TO_KAGGLE_NAMES inverse: Kaggle name → new-dataset name.
      2. Direct match: Kaggle TeamName == new-dataset TEAM (works for ~171 teams).

    Args:
        features_df: New dataset features (must have TEAM column).
        teams_df:    Kaggle MTeams DataFrame (TeamID, TeamName).

    Returns:
        Dict {kaggle_team_id: new_dataset_team_name}.
    """
    from src.features.coaching import CBB_TO_KAGGLE_NAMES

    new_team_set = set(features_df["TEAM"].unique())

    # CBB_TO_KAGGLE_NAMES: {new_name: kaggle_name}
    # Invert: {kaggle_name: new_name}, keeping only entries where new_name exists
    kaggle_name_to_new: dict[str, str] = {
        kag_name: new_name
        for new_name, kag_name in CBB_TO_KAGGLE_NAMES.items()
        if new_name in new_team_set
    }

    kaggle_to_new: dict[int, str] = {}
    for _, row in teams_df.iterrows():
        tid = int(row["TeamID"])
        kaggle_name = row["TeamName"]

        if kaggle_name in kaggle_name_to_new:
            kaggle_to_new[tid] = kaggle_name_to_new[kaggle_name]
        elif kaggle_name in new_team_set:
            kaggle_to_new[tid] = kaggle_name

    log.info(f"Name bridge: {len(kaggle_to_new)} Kaggle IDs mapped to new-dataset names")
    return kaggle_to_new


def build_matchup_data(
    features_df: pd.DataFrame,
    train_years: list[int] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Build symmetric binary matchup DataFrame using Kaggle game pairs + new features.

    For each actual tournament game (winner beats loser):
        row1: features(winner) - features(loser), LABEL=1
        row2: features(loser)  - features(winner), LABEL=0

    YEAR column is preserved for temporal CV splits.

    Args:
        features_df: Output of load_new_features() — one row per (YEAR, TEAM).
        train_years: Years to include. Defaults to 2008–2024 (excl. 2020).
        save:        If True, write to data/processed/matchups_new.csv.

    Returns:
        DataFrame with YEAR, LABEL, and one diff column per feature.
    """
    if train_years is None:
        train_years = [y for y in range(2008, 2025) if y != 2020]

    # Load Kaggle game results
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Kaggle results not found: {RESULTS_FILE}")
    if not TEAMS_FILE.exists():
        raise FileNotFoundError(f"Kaggle teams not found: {TEAMS_FILE}")

    results = pd.read_csv(RESULTS_FILE).rename(columns={"Season": "YEAR"})
    teams   = pd.read_csv(TEAMS_FILE)
    results = results[results["YEAR"].isin(train_years)]

    # Build Kaggle ID → new-dataset name bridge
    kaggle_to_new = _build_kaggle_to_new_name(features_df, teams)

    # Feature columns (all numeric, excluding identifiers)
    id_cols = {"YEAR", "TEAM", "SEED", "CONF", "ROUNDS_WON"}
    all_feat_cols = [
        c for c in features_df.columns
        if c not in id_cols and pd.api.types.is_numeric_dtype(features_df[c])
    ]

    # Drop features with >20% NaN in training data — keeps lookup coverage high
    train_df = features_df[features_df["YEAR"].isin(train_years)]
    null_frac = train_df[all_feat_cols].isna().mean()
    feat_cols = [c for c in all_feat_cols if null_frac[c] <= 0.20]
    dropped = set(all_feat_cols) - set(feat_cols)
    if dropped:
        log.info(f"Dropped {len(dropped)} high-null features (>20%): {sorted(dropped)}")

    # Impute remaining NaNs with column median (within training years only)
    col_medians = train_df[feat_cols].median()
    feat_df = features_df[features_df["YEAR"].isin(train_years)].copy()
    feat_df[feat_cols] = feat_df[feat_cols].fillna(col_medians)

    # Build feature lookup: (team_name, year) → np.ndarray
    feat_lookup: dict[tuple[str, int], np.ndarray] = {}
    for _, row in feat_df.iterrows():
        vals = row[feat_cols].values.astype(float)
        feat_lookup[(row["TEAM"], int(row["YEAR"]))] = vals

    records = []
    skipped = 0

    for _, row in results.iterrows():
        year   = int(row["YEAR"])
        w_name = kaggle_to_new.get(int(row["WTeamID"]))
        l_name = kaggle_to_new.get(int(row["LTeamID"]))

        if w_name is None or l_name is None:
            skipped += 1
            continue

        w_feats = feat_lookup.get((w_name, year))
        l_feats = feat_lookup.get((l_name, year))

        if w_feats is None or l_feats is None:
            skipped += 1
            continue

        diff = w_feats - l_feats
        records.append({"YEAR": year, "LABEL": 1, **dict(zip(feat_cols, diff))})
        records.append({"YEAR": year, "LABEL": 0, **dict(zip(feat_cols, -diff))})

    if skipped:
        log.warning(f"Skipped {skipped} games (unmapped team names or missing features)")

    df = pd.DataFrame(records)
    log.info(
        f"Matchup data: {len(df)//2} games, {df['YEAR'].nunique()} seasons "
        f"({len(feat_cols)} features, {skipped} games skipped)"
    )

    if save:
        out = PROCESSED_DIR / "matchups_new.csv"
        df.to_csv(out, index=False)
        log.info(f"Saved to {out}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from src.features.new_data_loader import load_new_features

    features = load_new_features(save=False)
    matchups = build_matchup_data(features)

    print(f"\nMatchup dataset: {len(matchups)//2} games, {matchups.shape[1]-2} features")
    print(f"Games per year:\n{matchups.groupby('YEAR').size().div(2).astype(int).to_string()}")

    # Sanity check: higher KADJ EM should win ~70%+ of games
    fav_wins = (matchups[matchups['LABEL']==1]['KADJ EM'] > 0).mean()
    print(f"\nSanity check — fraction where higher KADJ EM won: {fav_wins:.3f} (expect ~0.70)")

    print(f"\nNaN pct per feature (>5% shown):")
    feat_cols = [c for c in matchups.columns if c not in {'YEAR','LABEL'}]
    nan_pct = (matchups[feat_cols].isna().sum() / len(matchups) * 100).round(1)
    print(nan_pct[nan_pct > 5].sort_values(ascending=False).to_string())
