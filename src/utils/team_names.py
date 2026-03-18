"""
team_names.py — Shared utility for bridging Kaggle TeamID ↔ CBB team names.

Kaggle data uses numeric TeamIDs and its own spelling of team names.
CBB data (barttorvik.com) uses different spellings. The CBB_TO_KAGGLE_NAMES
map in coaching.py is the single source of truth for bridging them.

This module extracts _build_kaggle_to_cbb_map from win_probability.py
so it can be imported by formula_model.py, backtest.py, shap_selector.py,
and ui/app.py without depending on the old model file.
"""

import pandas as pd


def build_daynum_to_round(year_results: pd.DataFrame) -> dict[int, int]:
    """
    Infer DayNum → round mapping dynamically from game counts in a single year.

    NCAA tournament rounds have fixed game counts: 4 (First Four), 32 (R64),
    16 (R32), 8 (S16), 4 (E8), 2 (F4), 1 (Championship). DayNums shift by
    year so we infer the mapping from the chronological game-count pattern.

    Args:
        year_results: Results DataFrame filtered to a single season.

    Returns:
        Dict mapping DayNum → round number (0 = First Four, 6 = Champion).
    """
    ROUND_GAME_COUNTS = [4, 32, 16, 8, 4, 2, 1]  # rounds 0–6 in chronological order

    counts = year_results.groupby("DayNum").size().reset_index(name="n")
    days_sorted = sorted(counts["DayNum"].tolist())
    day_counts = {row["DayNum"]: row["n"] for _, row in counts.iterrows()}

    daynum_to_round: dict[int, int] = {}
    round_idx = 0
    accumulated = 0

    for day in days_sorted:
        n = day_counts[day]
        accumulated += n
        cumulative_target = sum(ROUND_GAME_COUNTS[: round_idx + 1])
        daynum_to_round[day] = round_idx
        if accumulated >= cumulative_target:
            round_idx += 1

    return daynum_to_round


def build_kaggle_to_cbb_map(features: pd.DataFrame, teams: pd.DataFrame) -> dict[int, str]:
    """
    Build Kaggle TeamID → CBB team name mapping using MTeams.csv as a bridge.

    Strategy:
      1. Try direct name match (Kaggle TeamName == CBB TEAM string).
      2. Fall back to CBB_TO_KAGGLE_NAMES inverse map in coaching.py.

    Args:
        features: CBB feature DataFrame with a TEAM column.
        teams:    MTeams DataFrame with TeamID and TeamName columns.

    Returns:
        Dict {kaggle_team_id: cbb_team_name} for all resolvable teams.
    """
    from src.features.coaching import CBB_TO_KAGGLE_NAMES

    kaggle_to_cbb = {v: k for k, v in CBB_TO_KAGGLE_NAMES.items()}
    cbb_team_set = set(features["TEAM"].unique())

    result: dict[int, str] = {}
    for _, row in teams.iterrows():
        kaggle_name = row["TeamName"]
        team_id = int(row["TeamID"])
        if kaggle_name in cbb_team_set:
            result[team_id] = kaggle_name
        elif kaggle_name in kaggle_to_cbb:
            cbb_name = kaggle_to_cbb[kaggle_name]
            if cbb_name in cbb_team_set:
                result[team_id] = cbb_name

    return result
