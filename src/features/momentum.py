"""
momentum.py — Compute Quality Momentum Score (QMS) for each team/year.

QMS = sum of weighted wins over the last 10 games before the tournament.
Weights are based on opponent's KenPom rank at game time:
  Win vs top-25 team  → 10 pts
  Win vs top-50 team  →  7 pts
  Win vs top-100 team →  4 pts
  Win vs below-100    →  1 pt
  Loss                →  0 pts

Raw win streaks are misleading — winning 8 of 10 against weak opponents
tells you nothing. Winning 6 of 10 against top-50 opponents tells you
the team is tournament-ready.

Flag: QMS > 60 historically correlates with outperforming seed.
"""

import logging
from pathlib import Path

import pandas as pd

from config import EXTERNAL_DIR, PROCESSED_DIR, QMS_WEIGHTS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TORVIK_DIR = EXTERNAL_DIR / "torvik"
LAST_N_GAMES = 10

# Tournament typically starts mid-March; cut off regular season by Mar 15
TOURNAMENT_CUTOFF = {"month": 3, "day": 15}


def _load_torvik_year(year: int) -> pd.DataFrame:
    """Load cached Torvik game data for one year."""
    path = TORVIK_DIR / f"torvik_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Torvik data not found for {year}. Run fetch_torvik.py first.")
    df = pd.read_csv(path, parse_dates=["DATE"])
    return df


def _opponent_rank(row: pd.Series, team: str) -> float:
    """
    Return the opponent's efficiency rank for a given team in a game row.

    Uses GAME_KEY to determine if team appears first or second,
    then returns the other team's rank.

    Args:
        row:  One row from the Torvik games DataFrame.
        team: The team we are computing QMS for.

    Returns:
        Opponent rank as float, or NaN if indeterminate.
    """
    game_key = str(row["GAME_KEY"])
    # Team appearing first in GAME_KEY corresponds to T1_RANK
    if game_key.startswith(team):
        return row["T2_RANK"]
    return row["T1_RANK"]


def _win_weight(opp_rank: float) -> int:
    """
    Map opponent rank to QMS win weight.

    Args:
        opp_rank: Opponent's national efficiency rank.

    Returns:
        Weight integer per SKILLS.md specification.
    """
    if pd.isna(opp_rank):
        return QMS_WEIGHTS["below_100"]
    if opp_rank <= 25:
        return QMS_WEIGHTS["top_25"]
    if opp_rank <= 50:
        return QMS_WEIGHTS["top_50"]
    if opp_rank <= 100:
        return QMS_WEIGHTS["top_100"]
    return QMS_WEIGHTS["below_100"]


def _team_games(games: pd.DataFrame, team: str) -> pd.DataFrame:
    """
    Filter games DataFrame to rows involving a specific team.
    Uses GAME_KEY and WINNER (both always populated) for matching.

    Args:
        games: Full Torvik games for one year.
        team:  Team name to filter for.

    Returns:
        DataFrame of games involving that team, sorted by DATE.
    """
    key_match = games["GAME_KEY"].astype(str).str.contains(team, na=False, regex=False)
    return games[key_match].sort_values("DATE")


def compute_qms(team_games: pd.DataFrame, team: str, cutoff_date: pd.Timestamp) -> float:
    """
    Compute Quality Momentum Score for one team.

    Takes the last N games before the tournament cutoff date and scores
    each win by opponent quality.

    Args:
        team_games:  DataFrame of games involving this team (sorted by DATE).
        team:        Team name string.
        cutoff_date: Games after this date are excluded (tournament starts).

    Returns:
        QMS as float. Returns 0.0 if no games found.
    """
    regular = team_games[team_games["DATE"] < cutoff_date]
    last_n = regular.tail(LAST_N_GAMES)

    if last_n.empty:
        return 0.0

    total = 0
    for _, row in last_n.iterrows():
        winner = str(row.get("WINNER", ""))
        if team.lower() not in winner.lower():
            continue  # loss
        opp_rank = _opponent_rank(row, team)
        total += _win_weight(opp_rank)

    return float(total)


def build_qms_features(
    teams_path: Path = PROCESSED_DIR / "features_efficiency.csv",
) -> pd.DataFrame:
    """
    Compute QMS for every tournament team in every available year.

    Args:
        teams_path: Path to the efficiency features CSV (has TEAM, YEAR, SEED).

    Returns:
        Original DataFrame with QMS column added.
    """
    df = pd.read_csv(teams_path)
    df["QMS"] = 0.0

    available_years = [
        y for y in df["YEAR"].unique()
        if (TORVIK_DIR / f"torvik_{y}.csv").exists()
    ]
    log.info(f"Computing QMS for years: {sorted(available_years)}")

    for year in sorted(available_years):
        games = _load_torvik_year(year)
        cutoff = pd.Timestamp(year=int(year), month=3, day=15)

        # Only compute for tournament teams (have a SEED)
        tourney_mask = df["YEAR"] == year
        teams = df.loc[tourney_mask & df["SEED"].notna(), "TEAM"].unique()

        scores = {}
        for team in teams:
            tgames = _team_games(games, team)
            scores[team] = compute_qms(tgames, team, cutoff)

        for team, score in scores.items():
            df.loc[(df["YEAR"] == year) & (df["TEAM"] == team), "QMS"] = score

        found = sum(1 for s in scores.values() if s > 0)
        log.info(f"{year}: QMS computed for {found}/{len(teams)} tournament teams")

    return df


def save(df: pd.DataFrame, filename: str = "features_momentum.csv") -> Path:
    """Save momentum features to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / filename
    df.to_csv(out, index=False)
    log.info(f"Saved to {out}")
    return out


if __name__ == "__main__":
    df = build_qms_features()
    save(df)

    tourney = df[
        df["SEED"].notna() &
        (df["YEAR"] != 2020) &
        (df["YEAR"] != 2025) &
        (df["QMS"] > 0)
    ]

    print(f"\nQMS range: {tourney['QMS'].min():.0f} to {tourney['QMS'].max():.0f}")
    print(f"\nTop 10 QMS (tournament teams):")
    print(
        tourney.nlargest(10, "QMS")[["YEAR", "TEAM", "SEED", "QMS"]]
        .to_string(index=False)
    )
    print(f"\nTeams with QMS > 60 (historically overperform seed):")
    hot = tourney[tourney["QMS"] > 60]
    print(f"  Count: {len(hot)}")
    print(hot[["YEAR", "TEAM", "SEED", "QMS"]].to_string(index=False))
