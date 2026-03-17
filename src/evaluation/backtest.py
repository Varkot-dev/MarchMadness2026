"""
backtest.py — Historical bracket prediction backtest.

For each target year from 2016–2024 (skip 2020):
  1. Train win probability model on all seasons prior to target year.
  2. Build win probability matrix for that year's tournament teams.
  3. Run Monte Carlo simulation (10k runs) using that year's bracket structure.
  4. Score best predicted bracket against actual tournament results.
  5. Report ESPN score, champion prediction accuracy, and round-by-round accuracy.

Usage:
    python -m src.evaluation.backtest --year 2024
    python -m src.evaluation.backtest --all
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import EXTERNAL_DIR, PROCESSED_DIR
from src.models.simulator import (
    build_bracket_from_seeds,
    run_simulations,
    score_bracket,
    ESPN_ROUND_POINTS,
)
from src.models.win_probability import (
    DIFF_FEATURES,
    train_logistic,
    train_xgboost_margin,
    margin_to_prob,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

KAGGLE_DIR = EXTERNAL_DIR / "kaggle"
SEEDS_FILE = KAGGLE_DIR / "MNCAATourneySeeds.csv"
RESULTS_FILE = KAGGLE_DIR / "MNCAATourneyDetailedResults.csv"
TEAMS_FILE = KAGGLE_DIR / "MTeams.csv"
FEATURES_FILE = PROCESSED_DIR / "features_coaching.csv"

# DayNum ranges → tournament round number.
# Round 0 = First Four (winners advance to round 1).
# DayNum → round mapping verified from 2019–2024 Kaggle data:
#   134, 135      → 0  First Four   (4 games, 2/day)
#   136–140       → 1  R64          (32 games, 16/day across 2 days)
#   138–140       → 1  R64 cont.    (some years stagger days)
#   143–144       → 2  R32          (16 games, 8/day — standard years)
#   145–148       → 2  R32 cont.    (2021 bubble used different days)
#   145–146       → 3  S16          (8 games, 4/day — standard years)
#   147–148       → 3  S16 cont.    (bubble years)
#   152           → 4  E8 / F4 pre  — but 2 games = F4 semis in 2024
# Observed counts 2024: 136=16,137=16,138=8,139=8,143=4,144=4,145=2,146=2,152=2,154=1
# This gives: R64=136-137, R32=138-139, S16=143-144, E8=145-146, F4=152, Champ=154
DAYNUM_TO_ROUND: dict[int, int] = {
    134: 0, 135: 0,                          # First Four
    136: 1, 137: 1, 138: 1, 139: 1, 140: 1, # R64
    141: 2, 142: 2, 143: 2, 144: 2,          # R32
    145: 3, 146: 3, 147: 3, 148: 3,          # S16
    149: 4, 150: 4, 151: 4, 152: 4,          # E8 (incl. F4 in bubble 2021)
    153: 5, 154: 5, 155: 5,                  # F4
    156: 6, 157: 6, 158: 6, 159: 6,          # Championship
    160: 6, 161: 6, 162: 6, 163: 6,
}

BACKTEST_YEARS = [y for y in range(2016, 2025) if y != 2020]


# ── Actual results loading ─────────────────────────────────────────────────────

def _build_daynum_to_round(year_results: pd.DataFrame) -> dict[int, int]:
    """
    Infer DayNum → round mapping dynamically from game counts in a single year.

    NCAA tournament rounds have fixed game counts: 4 (First Four), 32 (R64),
    16 (R32), 8 (S16), 4 (E8), 2 (F4), 1 (Championship). DayNums vary by
    year so we sort DayNums by ascending game count and assign rounds 0–6.

    Args:
        year_results: Results DataFrame filtered to a single season.

    Returns:
        Dict mapping DayNum → round number (0 = First Four, 6 = Champion).
    """
    # Games per day, sorted ascending (fewest = later rounds)
    counts = year_results.groupby("DayNum").size().reset_index(name="n")
    # Sort: first by game count descending (R64 first), then by DayNum ascending
    counts = counts.sort_values(["n", "DayNum"], ascending=[False, True])

    # Expected game counts in order: First Four=4, R64=32, R32=16, S16=8, E8=4, F4=2, Champ=1
    # But First Four (4 games) and E8 (4 games) have the same count — distinguish by DayNum
    # Strategy: assign rounds by DayNum order within game-count groups.
    # Simpler: assign round based on cumulative game position.
    # We know rounds 0–6 have game counts [4, 32, 16, 8, 4, 2, 1] in chronological order.
    ROUND_GAME_COUNTS = [4, 32, 16, 8, 4, 2, 1]  # rounds 0–6

    # Sort all games by DayNum ascending to get chronological order
    days_sorted = sorted(counts["DayNum"].tolist())
    day_counts = {row["DayNum"]: row["n"] for _, row in counts.iterrows()}

    daynum_to_round: dict[int, int] = {}
    round_idx = 0
    accumulated = 0

    for day in days_sorted:
        n = day_counts[day]
        accumulated += n
        # Determine which round this day belongs to by cumulative game count
        cumulative_target = sum(ROUND_GAME_COUNTS[:round_idx + 1])
        daynum_to_round[day] = round_idx
        if accumulated >= cumulative_target:
            round_idx += 1

    return daynum_to_round


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
        results_df:     MNCAATourneyDetailedResults DataFrame.
        teams_df:       MTeams DataFrame (TeamID, TeamName).
        kaggle_to_cbb:  Kaggle TeamID → CBB team name mapping.
        year:           Tournament season year.

    Returns:
        Dict mapping CBB team name → furthest round reached (0–6).
        Teams that lost in the First Four show round 0; round 6 = champion.
    """
    year_results = results_df[results_df["Season"] == year].copy()
    if year_results.empty:
        log.warning(f"No results found for year {year}")
        return {}

    daynum_to_round = _build_daynum_to_round(year_results)

    furthest: dict[str, int] = {}

    for _, row in year_results.iterrows():
        day_num = int(row["DayNum"])
        rnd = daynum_to_round.get(day_num)
        if rnd is None:
            log.warning(f"Unrecognised DayNum {day_num} in year {year} — skipping")
            continue

        winner_id = int(row["WTeamID"])
        loser_id = int(row["LTeamID"])
        winner_name = kaggle_to_cbb.get(winner_id)
        loser_name = kaggle_to_cbb.get(loser_id)

        if winner_name is None:
            log.warning(f"Unmapped TeamID {winner_id} in year {year}")
        else:
            furthest[winner_name] = max(furthest.get(winner_name, 0), rnd)

        if loser_name is None:
            log.warning(f"Unmapped TeamID {loser_id} in year {year}")
        else:
            loser_round = max(rnd - 1, 0)
            if loser_name not in furthest:
                furthest[loser_name] = loser_round

    log.info(
        f"Actual results loaded for {year}: "
        f"{len(furthest)} teams, champion={_find_champion(furthest)}"
    )
    return furthest


def _find_champion(results: dict[str, int]) -> str:
    """Return the team with round 6 (Champion) from a results dict."""
    for team, rnd in results.items():
        if rnd == 6:
            return team
    return "Unknown"


# ── Win probability matrix for a target year ──────────────────────────────────

def build_win_prob_matrix(
    features_df: pd.DataFrame,
    target_year: int,
    kaggle_to_cbb: dict[int, str] | None = None,
    matchup_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Train a win probability model on seasons prior to target_year and build
    a square probability matrix for all tournament teams in target_year.

    Training uses only seasons strictly before target_year to prevent leakage.
    Teams present in the features file for target_year with a non-null SEED
    column are treated as tournament teams.

    Args:
        features_df:   Full features DataFrame (all years, from features_coaching.csv).
                       Used to look up each team's raw feature vector for the matrix.
        target_year:   Year to build matrix for.
        kaggle_to_cbb: Optional Kaggle TeamID → CBB name map (accepted for API
                       consistency with callers; not used directly here).
        matchup_df:    Symmetric matchup DataFrame from load_matchup_data() with
                       LABEL, MARGIN, and DIFF_FEATURES columns. Used for training.
                       If None, load_matchup_data() is called internally.

    Returns:
        Square DataFrame indexed by team name with P(row beats col) values.
        Returns an empty DataFrame if fewer than 2 tournament teams are found.
    """
    # Tournament teams for target year
    tourney = features_df[
        (features_df["YEAR"] == target_year) & features_df["SEED"].notna()
    ][["TEAM"] + DIFF_FEATURES].dropna(subset=DIFF_FEATURES)

    teams = tourney["TEAM"].tolist()
    if len(teams) < 2:
        log.warning(
            f"Fewer than 2 tournament teams found for {target_year} — "
            "cannot build matrix"
        )
        return pd.DataFrame()

    log.info(
        f"Building win prob matrix for {target_year}: {len(teams)} teams"
    )

    # Training data: symmetric matchup rows (LABEL + MARGIN) from prior years
    if matchup_df is None:
        from src.models.win_probability import load_matchup_data
        matchup_df = load_matchup_data()

    train_df = matchup_df[matchup_df["YEAR"] < target_year].dropna(
        subset=DIFF_FEATURES + ["LABEL"]
    )

    if len(train_df) < 50:
        log.warning(
            f"Only {len(train_df)} training rows before {target_year} — "
            "skipping matrix build"
        )
        return pd.DataFrame()

    X_train = train_df[DIFF_FEATURES].values
    y_label = train_df["LABEL"].values
    y_margin = train_df["MARGIN"].values if "MARGIN" in train_df.columns else None

    lr_model, scaler = train_logistic(X_train, y_label)

    xgb_model = None
    if y_margin is not None:
        try:
            xgb_model = train_xgboost_margin(X_train, y_margin)
        except Exception as e:
            log.warning(f"XGBoost training failed for {target_year}: {e}")

    # Build pairwise probability matrix
    matrix = pd.DataFrame(np.nan, index=teams, columns=teams)
    for i, team_a in enumerate(teams):
        a_feats = tourney[tourney["TEAM"] == team_a][DIFF_FEATURES].values[0]
        for j, team_b in enumerate(teams):
            if i == j:
                matrix.loc[team_a, team_b] = 0.5
                continue
            b_feats = tourney[tourney["TEAM"] == team_b][DIFF_FEATURES].values[0]
            diff = (a_feats - b_feats).reshape(1, -1)

            lr_prob = lr_model.predict_proba(scaler.transform(diff))[0, 1]

            if xgb_model is not None:
                xgb_prob = margin_to_prob(xgb_model.predict(diff))[0]
                prob = (lr_prob + xgb_prob) / 2.0
            else:
                prob = lr_prob

            matrix.loc[team_a, team_b] = round(float(prob), 4)

    log.info(
        f"Win prob matrix complete for {target_year}: "
        f"{len(teams)}x{len(teams)}, "
        f"{'LR+XGB ensemble' if xgb_model else 'LR only'}"
    )
    return matrix


# ── Per-year backtest ──────────────────────────────────────────────────────────

def run_backtest_year(
    year: int,
    seeds_df: pd.DataFrame,
    results_df: pd.DataFrame,
    features_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    matchup_df: pd.DataFrame,
    kaggle_to_cbb: dict[int, str],
    n_sims: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Run full backtest pipeline for a single tournament year.

    Steps:
      1. Build win probability matrix (trained on prior years only).
      2. Build bracket structure from that year's seeds.
      3. Run Monte Carlo simulation.
      4. Score best predicted bracket against actual results.
      5. Compute round-by-round accuracy.

    Args:
        year:          Target tournament year.
        seeds_df:      MNCAATourneySeeds DataFrame.
        results_df:    MNCAATourneyDetailedResults DataFrame.
        features_df:   features_coaching DataFrame (all years).
        teams_df:      MTeams DataFrame.
        matchup_df:    Symmetric matchup DataFrame from load_matchup_data()
                       (used to supply LABEL and MARGIN for win prob training).
        kaggle_to_cbb: Kaggle TeamID → CBB name map.
        n_sims:        Number of Monte Carlo simulations.
        seed:          Random seed for reproducibility.

    Returns:
        Dict with keys: year, predicted_champion, actual_champion,
        espn_score, max_possible_score, round_accuracy (dict[int, float]).
        Returns a dict with year and an error key if build fails.
    """
    log.info(f"=== Backtest year: {year} ===")

    # Build win prob matrix — features for team vectors, matchup_df for training
    prob_matrix = build_win_prob_matrix(features_df, year, kaggle_to_cbb, matchup_df)
    if prob_matrix.empty:
        log.warning(f"Skipping {year} — could not build prob matrix")
        return {"year": year, "error": "no_prob_matrix"}

    # Build bracket structure
    try:
        bracket_structure = build_bracket_from_seeds(seeds_df, teams_df, year)
    except ValueError as exc:
        log.warning(f"Skipping {year} — bracket build failed: {exc}")
        return {"year": year, "error": str(exc)}

    # Run simulations
    sim_results = run_simulations(
        prob_matrix, bracket_structure, n_sims=n_sims, seed=seed
    )
    best_bracket = sim_results["p90_bracket"]

    # Load actual results
    actual = load_actual_results(results_df, teams_df, kaggle_to_cbb, year)
    if not actual:
        log.warning(f"Skipping {year} — no actual results available")
        return {"year": year, "error": "no_actual_results"}

    predicted_champion = best_bracket["champion"]
    actual_champion = _find_champion(actual)

    # ESPN score
    espn_score = score_bracket(best_bracket["picks"], actual)

    # Max possible score: sum of all points if every pick were correct.
    # Upper bound = picking every game correctly in a 64-team bracket.
    # Round N has 2^(6-N) games × ESPN_ROUND_POINTS[N].
    max_possible = sum(
        (2 ** (6 - rnd)) * pts for rnd, pts in ESPN_ROUND_POINTS.items()
    )

    # Round-by-round pick accuracy
    # For each round, count: teams predicted to reach that round vs how many did.
    round_accuracy: dict[int, float] = {}
    for rnd in range(1, 7):
        predicted_teams = {t for t, r in best_bracket["picks"].items() if r >= rnd}
        actual_teams = {t for t, r in actual.items() if r >= rnd}

        if not predicted_teams:
            round_accuracy[rnd] = float("nan")
            continue

        correct = len(predicted_teams & actual_teams)
        round_accuracy[rnd] = round(correct / len(predicted_teams), 4)

    log.info(
        f"Year {year}: predicted={predicted_champion}, actual={actual_champion}, "
        f"ESPN score={espn_score}/{max_possible}, "
        f"R1 acc={round_accuracy.get(1, 0):.1%}"
    )

    return {
        "year": year,
        "predicted_champion": predicted_champion,
        "actual_champion": actual_champion,
        "champion_correct": predicted_champion == actual_champion,
        "espn_score": espn_score,
        "max_possible_score": max_possible,
        "pct_max": round(espn_score / max_possible, 4) if max_possible else 0.0,
        "round_accuracy": round_accuracy,
    }


# ── Full backtest ──────────────────────────────────────────────────────────────

def run_full_backtest(
    years: list[int],
    seeds_df: pd.DataFrame,
    results_df: pd.DataFrame,
    features_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    matchup_df: pd.DataFrame,
    kaggle_to_cbb: dict[int, str],
    n_sims: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run backtest for all specified years and return a summary DataFrame.

    Each row represents one tournament year with model performance metrics.

    Args:
        years:         List of target years to evaluate.
        seeds_df:      MNCAATourneySeeds DataFrame.
        results_df:    MNCAATourneyDetailedResults DataFrame.
        features_df:   features_coaching DataFrame (all years).
        teams_df:      MTeams DataFrame.
        matchup_df:    Symmetric matchup DataFrame (provides LABEL/MARGIN).
        kaggle_to_cbb: Kaggle TeamID → CBB name map.
        n_sims:        Monte Carlo simulations per year.
        seed:          Random seed base (each year offset by year index).

    Returns:
        DataFrame with one row per successfully evaluated year, columns:
        year, predicted_champion, actual_champion, champion_correct,
        espn_score, max_possible_score, pct_max, r1_acc, r2_acc, ..., r6_acc.
    """
    all_records: list[dict] = []

    for idx, year in enumerate(years):
        result = run_backtest_year(
            year=year,
            seeds_df=seeds_df,
            results_df=results_df,
            features_df=features_df,
            teams_df=teams_df,
            matchup_df=matchup_df,
            kaggle_to_cbb=kaggle_to_cbb,
            n_sims=n_sims,
            seed=seed + idx,  # vary seed per year for independence
        )

        if "error" in result:
            log.warning(f"Year {year} skipped: {result['error']}")
            continue

        # Flatten round_accuracy into separate columns
        flat = {k: v for k, v in result.items() if k != "round_accuracy"}
        for rnd, acc in result.get("round_accuracy", {}).items():
            flat[f"r{rnd}_acc"] = acc

        all_records.append(flat)

    if not all_records:
        log.error("No years successfully evaluated — returning empty DataFrame")
        return pd.DataFrame()

    summary = pd.DataFrame(all_records)
    log.info(
        f"Backtest complete: {len(summary)} years evaluated\n"
        f"  Mean ESPN score: {summary['espn_score'].mean():.1f} "
        f"/ {summary['max_possible_score'].iloc[0]}\n"
        f"  Champion accuracy: "
        f"{summary['champion_correct'].mean():.1%} "
        f"({summary['champion_correct'].sum()}/{len(summary)})"
    )
    return summary


# ── Data loading helpers ───────────────────────────────────────────────────────

def _load_all_data() -> tuple:
    """
    Load all required DataFrames and the Kaggle→CBB name map.

    Returns:
        Tuple of (seeds_df, results_df, features_df, teams_df, matchup_df,
                  kaggle_to_cbb).
    """
    for path in (SEEDS_FILE, RESULTS_FILE, TEAMS_FILE, FEATURES_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run the data pipeline first."
            )

    seeds_df = pd.read_csv(SEEDS_FILE)
    results_df = pd.read_csv(RESULTS_FILE)
    features_df = pd.read_csv(FEATURES_FILE)
    teams_df = pd.read_csv(TEAMS_FILE)

    # Build Kaggle TeamID → CBB name map (reuse logic from win_probability.py)
    from src.models.win_probability import _build_kaggle_to_cbb_map
    kaggle_to_cbb = _build_kaggle_to_cbb_map(features_df, teams_df)

    # Build symmetric matchup DataFrame for training (provides LABEL, MARGIN)
    from src.models.win_probability import load_matchup_data
    matchup_df = load_matchup_data(RESULTS_FILE, FEATURES_FILE, TEAMS_FILE)

    log.info(
        f"Data loaded: {len(seeds_df)} seed rows, "
        f"{len(results_df)} result rows, "
        f"{len(features_df)} feature rows, "
        f"{len(matchup_df)} matchup rows"
    )
    return seeds_df, results_df, features_df, teams_df, matchup_df, kaggle_to_cbb


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run historical bracket prediction backtest."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--year",
        type=int,
        help="Single year to backtest (e.g. 2024).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help=f"Run backtest for all years: {BACKTEST_YEARS}.",
    )
    args = parser.parse_args()

    seeds_df, results_df, features_df, teams_df, matchup_df, kaggle_to_cbb = (
        _load_all_data()
    )

    if args.all:
        years_to_run = BACKTEST_YEARS
    else:
        if args.year not in BACKTEST_YEARS:
            log.warning(
                f"Year {args.year} is not in the standard backtest range "
                f"({BACKTEST_YEARS}) — attempting anyway."
            )
        years_to_run = [args.year]

    summary = run_full_backtest(
        years=years_to_run,
        seeds_df=seeds_df,
        results_df=results_df,
        features_df=features_df,
        teams_df=teams_df,
        matchup_df=matchup_df,
        kaggle_to_cbb=kaggle_to_cbb,
        n_sims=10_000,
        seed=42,
    )

    if summary.empty:
        print("No results to display.")
    else:
        round_cols = [c for c in summary.columns if c.startswith("r") and c.endswith("_acc")]
        display_cols = [
            "year", "predicted_champion", "actual_champion", "champion_correct",
            "espn_score", "max_possible_score", "pct_max",
        ] + round_cols

        print("\n" + "=" * 90)
        print("Backtest Summary")
        print("=" * 90)
        print(summary[display_cols].to_string(index=False))

        print("\n" + "=" * 90)
        print("Aggregate Statistics")
        print("=" * 90)
        print(f"  Years evaluated:       {len(summary)}")
        print(f"  Mean ESPN score:       {summary['espn_score'].mean():.1f}")
        print(f"  Median ESPN score:     {summary['espn_score'].median():.1f}")
        print(f"  Best ESPN score:       {summary['espn_score'].max():.0f} "
              f"({summary.loc[summary['espn_score'].idxmax(), 'year']})")
        print(f"  Worst ESPN score:      {summary['espn_score'].min():.0f} "
              f"({summary.loc[summary['espn_score'].idxmin(), 'year']})")
        print(f"  Champion accuracy:     "
              f"{summary['champion_correct'].mean():.1%} "
              f"({summary['champion_correct'].sum()}/{len(summary)})")
        for rnd, name in [(1, "R64"), (2, "R32"), (3, "S16"), (4, "E8"),
                          (5, "F4"), (6, "Champ")]:
            col = f"r{rnd}_acc"
            if col in summary.columns:
                print(
                    f"  Mean {name} pick acc:  "
                    f"{summary[col].mean():.1%}"
                )
