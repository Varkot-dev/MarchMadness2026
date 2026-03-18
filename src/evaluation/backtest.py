"""
backtest.py — Historical bracket prediction backtest.

Runs formula_model.py's pipeline on holdout years to measure ESPN score,
champion prediction accuracy, and round-by-round accuracy.

All win probability logic lives in formula_model.py. This module provides:
  - _build_daynum_to_round()  — dynamic DayNum → round mapping (used everywhere)
  - load_actual_results()     — build team → furthest round reached from Kaggle data
  - run_backtest_year()       — evaluate one year via formula_model.simulate_bracket()
  - run_full_backtest()       — evaluate multiple years, return summary DataFrame

Usage:
    python -m src.evaluation.backtest --year 2024
    python -m src.evaluation.backtest --all
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import EXTERNAL_DIR, PROCESSED_DIR, ESPN_ROUND_POINTS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

KAGGLE_DIR   = EXTERNAL_DIR / "kaggle"
RESULTS_FILE = KAGGLE_DIR / "MNCAATourneyDetailedResults.csv"
TEAMS_FILE   = KAGGLE_DIR / "MTeams.csv"

BACKTEST_YEARS = [y for y in range(2016, 2025) if y != 2020]


# ── Round mapping ──────────────────────────────────────────────────────────────

def _build_daynum_to_round(year_results: pd.DataFrame) -> dict[int, int]:
    """
    Infer DayNum → round mapping dynamically from game counts in a single year.

    NCAA tournament rounds have fixed game counts: 4 (First Four), 32 (R64),
    16 (R32), 8 (S16), 4 (E8), 2 (F4), 1 (Championship). DayNums shift by
    year so we infer the mapping from the chronological game-count pattern
    rather than hardcoding values that break across years.

    Args:
        year_results: Results DataFrame filtered to a single season.

    Returns:
        Dict mapping DayNum → round number (0 = First Four, 6 = Champion).
    """
    from src.utils.team_names import build_daynum_to_round
    return build_daynum_to_round(year_results)


# ── Actual results ─────────────────────────────────────────────────────────────

def _find_champion(results: dict[str, int]) -> str:
    """Return the team with round 6 (Champion) from a results dict."""
    for team, rnd in results.items():
        if rnd == 6:
            return team
    return "Unknown"


def load_actual_results(
    results_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    kaggle_to_cbb: dict[int, str],
    year: int,
) -> dict[str, int]:
    """
    Build a team_name → furthest_round_reached mapping from actual results.

    Rounds are inferred dynamically from game counts per day (not hardcoded
    DayNum values) so this works correctly across all years including the
    2021 bubble tournament.

    Args:
        results_df:    MNCAATourneyDetailedResults DataFrame.
        teams_df:      MTeams DataFrame (TeamID, TeamName).
        kaggle_to_cbb: Kaggle TeamID → CBB team name mapping.
        year:          Tournament season year.

    Returns:
        Dict mapping CBB team name → furthest round reached (0–6).
        Teams that lost in the First Four show round 0; round 6 = champion.
    """
    from src.utils.team_names import build_daynum_to_round

    year_results = results_df[results_df["Season"] == year].copy()
    if year_results.empty:
        log.warning(f"No results found for year {year}")
        return {}

    daynum_to_round = build_daynum_to_round(year_results)
    furthest: dict[str, int] = {}

    for _, row in year_results.iterrows():
        rnd = daynum_to_round.get(int(row["DayNum"]))
        if rnd is None:
            continue
        winner_name = kaggle_to_cbb.get(int(row["WTeamID"]))
        loser_name  = kaggle_to_cbb.get(int(row["LTeamID"]))
        if winner_name:
            furthest[winner_name] = max(furthest.get(winner_name, 0), rnd)
        if loser_name and loser_name not in furthest:
            furthest[loser_name] = max(rnd - 1, 0)

    log.info(
        f"Actual results loaded for {year}: "
        f"{len(furthest)} teams, champion={_find_champion(furthest)}"
    )
    return furthest


# ── Per-year backtest ──────────────────────────────────────────────────────────

def run_backtest_year(year: int, n_sims: int = 10_000, seed: int = 42) -> dict:
    """
    Run the full backtest pipeline for a single tournament year.

    Delegates to formula_model.simulate_bracket() which trains strictly on
    prior years (no temporal leakage), then scores against actual results.

    Args:
        year:   Target tournament year.
        n_sims: Unused — kept for API compatibility. formula_model uses
                deterministic bracket simulation, not Monte Carlo per year.
        seed:   Unused — kept for API compatibility.

    Returns:
        Dict with: year, predicted_champion, actual_champion, champion_correct,
        espn_score, round_accuracy (dict[int, float]).
    """
    from src.models.formula_model import (
        load_matchup_data, fit_model, simulate_bracket,
        TRAIN_YEARS, run_temporal_cv,
    )

    log.info(f"=== Backtest year: {year} ===")

    # Train on all years strictly before this one (excl. 2020)
    df = load_matchup_data()
    train_years = [y for y in sorted(df["YEAR"].unique()) if y < year and y != 2020]
    if len(train_years) < 3:
        return {"year": year, "error": f"only {len(train_years)} prior seasons"}

    train_df = df[df["YEAR"].isin(train_years)]
    _, best_c = run_temporal_cv(train_df)
    model, scaler, _ = fit_model(train_df, c=best_c)

    result = simulate_bracket(year, model, scaler)
    if result["espn_score"] is None:
        return {"year": year, "error": "no_actual_results"}

    from src.models.formula_model import compute_round_accuracy
    round_acc = compute_round_accuracy(result["matchups"])

    actual_champ = _find_champion(result["actual"])
    log.info(
        f"Year {year}: predicted={result['champion']}, actual={actual_champ}, "
        f"ESPN={result['espn_score']}"
    )

    return {
        "year":                year,
        "predicted_champion":  result["champion"],
        "actual_champion":     actual_champ,
        "champion_correct":    result["champion"] == actual_champ,
        "espn_score":          result["espn_score"],
        "round_accuracy":      round_acc,
    }


# ── Full backtest ──────────────────────────────────────────────────────────────

def run_full_backtest(years: list[int] | None = None) -> pd.DataFrame:
    """
    Run backtest for multiple years, return a summary DataFrame.

    Args:
        years: Years to evaluate. Defaults to BACKTEST_YEARS (2016–2024, excl. 2020).

    Returns:
        DataFrame with one row per year: year, predicted_champion, actual_champion,
        champion_correct, espn_score, r1_acc … r6_acc.
    """
    if years is None:
        years = BACKTEST_YEARS

    records = []
    for year in years:
        result = run_backtest_year(year)
        if "error" in result:
            log.warning(f"Year {year} skipped: {result['error']}")
            continue
        flat = {k: v for k, v in result.items() if k != "round_accuracy"}
        for rnd, acc in result.get("round_accuracy", {}).items():
            flat[f"r{rnd}_acc"] = acc
        records.append(flat)

    if not records:
        log.error("No years successfully evaluated.")
        return pd.DataFrame()

    summary = pd.DataFrame(records)
    log.info(
        f"Backtest complete: {len(summary)} years | "
        f"Mean ESPN: {summary['espn_score'].mean():.1f} | "
        f"Champion acc: {summary['champion_correct'].mean():.1%}"
    )
    return summary


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run historical bracket backtest.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--year", type=int, help="Single year (e.g. 2024).")
    group.add_argument("--all", action="store_true",
                       help=f"All years: {BACKTEST_YEARS}.")
    args = parser.parse_args()

    years_to_run = BACKTEST_YEARS if args.all else [args.year]
    summary = run_full_backtest(years_to_run)

    if summary.empty:
        print("No results.")
    else:
        ROUND_NAMES = {1: "R64", 2: "R32", 3: "S16", 4: "E8", 5: "F4", 6: "Champ"}
        max_espn = sum(ESPN_ROUND_POINTS[r] * (2 ** (6 - r)) for r in range(1, 7))

        print("\n" + "=" * 80)
        print("BACKTEST SUMMARY")
        print("=" * 80)
        display_cols = ["year", "predicted_champion", "actual_champion",
                        "champion_correct", "espn_score"]
        print(summary[display_cols].to_string(index=False))

        print("\n" + "=" * 80)
        print(f"  Years evaluated:   {len(summary)}")
        print(f"  Mean ESPN score:   {summary['espn_score'].mean():.1f} / {max_espn}")
        print(f"  Best:              {summary['espn_score'].max():.0f} "
              f"({summary.loc[summary['espn_score'].idxmax(), 'year']})")
        print(f"  Worst:             {summary['espn_score'].min():.0f} "
              f"({summary.loc[summary['espn_score'].idxmin(), 'year']})")
        print(f"  Champion accuracy: "
              f"{summary['champion_correct'].mean():.1%} "
              f"({summary['champion_correct'].sum()}/{len(summary)})")
        for rnd, name in ROUND_NAMES.items():
            col = f"r{rnd}_acc"
            if col in summary.columns:
                print(f"  Mean {name} acc:     {summary[col].mean():.1%}")
