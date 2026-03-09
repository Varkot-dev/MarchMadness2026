"""
tests/test_momentum.py — Unit and sanity tests for src/features/momentum.py

Covers:
  - _win_weight(): all 4 rank buckets, NaN rank, boundary ranks
  - compute_qms(): synthetic DataFrames for all-wins, all-losses, mixed,
                   last-10-games window, and cutoff_date enforcement
  - Historical sanity checks against data/processed/features_momentum.csv
"""

import numpy as np
import pandas as pd
import pytest

from src.features.momentum import _win_weight, compute_qms
from config import QMS_WEIGHTS, PROCESSED_DIR


# ── _win_weight ───────────────────────────────────────────────────────────────

class TestWinWeight:
    """Test _win_weight() mapping of opponent rank → point value."""

    # --- Core buckets ---

    def test_top_25_bucket(self):
        """A rank of 10 is clearly inside top-25 → should return 10."""
        assert _win_weight(10) == QMS_WEIGHTS["top_25"]

    def test_top_50_bucket(self):
        """A rank of 35 is top-26 through top-50 → should return 7."""
        assert _win_weight(35) == QMS_WEIGHTS["top_50"]

    def test_top_100_bucket(self):
        """A rank of 75 is top-51 through top-100 → should return 4."""
        assert _win_weight(75) == QMS_WEIGHTS["top_100"]

    def test_below_100_bucket(self):
        """A rank of 200 is outside top-100 → should return 1."""
        assert _win_weight(200) == QMS_WEIGHTS["below_100"]

    # --- NaN rank ---

    def test_nan_rank_returns_below_100_weight(self):
        """NaN opponent rank (unknown) is treated as below-100 → 1 pt."""
        assert _win_weight(float("nan")) == QMS_WEIGHTS["below_100"]

    def test_numpy_nan_rank(self):
        """numpy NaN is also recognised as NaN by pd.isna → 1 pt."""
        assert _win_weight(np.nan) == QMS_WEIGHTS["below_100"]

    # --- Boundary ranks ---

    def test_boundary_rank_25_is_top_25(self):
        """Rank exactly 25 is the top boundary of the top-25 bucket → 10 pts."""
        assert _win_weight(25) == QMS_WEIGHTS["top_25"]

    def test_boundary_rank_26_is_top_50(self):
        """Rank 26 falls into the top-50 bucket → 7 pts."""
        assert _win_weight(26) == QMS_WEIGHTS["top_50"]

    def test_boundary_rank_50_is_top_50(self):
        """Rank exactly 50 is the top boundary of the top-50 bucket → 7 pts."""
        assert _win_weight(50) == QMS_WEIGHTS["top_50"]

    def test_boundary_rank_51_is_top_100(self):
        """Rank 51 falls into the top-100 bucket → 4 pts."""
        assert _win_weight(51) == QMS_WEIGHTS["top_100"]

    def test_boundary_rank_100_is_top_100(self):
        """Rank exactly 100 is the top boundary of the top-100 bucket → 4 pts."""
        assert _win_weight(100) == QMS_WEIGHTS["top_100"]

    def test_boundary_rank_101_is_below_100(self):
        """Rank 101 falls into the below-100 bucket → 1 pt."""
        assert _win_weight(101) == QMS_WEIGHTS["below_100"]


# ── compute_qms ───────────────────────────────────────────────────────────────

def _make_game(date: str, team: str, winner: str, t1_rank: float, t2_rank: float) -> dict:
    """
    Build a single-row game record matching the Torvik DataFrame schema.

    GAME_KEY is constructed as '<team>_vs_<opponent>' so that _opponent_rank()
    can extract the correct side's rank.  `team` is placed first so
    game_key.startswith(team) == True and T2_RANK is the opponent rank.
    """
    return {
        "DATE": pd.Timestamp(date),
        "GAME_KEY": f"{team}_vs_opponent",
        "WINNER": winner,
        "T1_RANK": t1_rank,
        "T2_RANK": t2_rank,
    }


def _build_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values("DATE").reset_index(drop=True)


class TestComputeQMS:
    """Unit tests for compute_qms() using small synthetic DataFrames."""

    TEAM = "TestTeam"
    CUTOFF = pd.Timestamp("2024-03-15")

    # Helper: team always listed first in GAME_KEY → opponent rank = T2_RANK.

    def _win_row(self, date: str, opp_rank: float) -> dict:
        return _make_game(date, self.TEAM, self.TEAM, t1_rank=999, t2_rank=opp_rank)

    def _loss_row(self, date: str, opp_rank: float) -> dict:
        return _make_game(date, self.TEAM, "Opponent", t1_rank=999, t2_rank=opp_rank)

    # --- All wins vs top-25 opponents → maximum possible score (10 × 10 = 100) ---

    def test_all_wins_vs_top25_gives_max_score(self):
        """10 wins over rank-1 opponents should produce 10 × 10 = 100 pts."""
        rows = [self._win_row(f"2024-02-{d:02d}", 1) for d in range(1, 11)]
        df = _build_df(rows)
        result = compute_qms(df, self.TEAM, self.CUTOFF)
        assert result == pytest.approx(100.0)

    # --- All losses → 0 ---

    def test_all_losses_gives_zero(self):
        """10 losses regardless of opponent quality → QMS = 0."""
        rows = [self._loss_row(f"2024-02-{d:02d}", 1) for d in range(1, 11)]
        df = _build_df(rows)
        result = compute_qms(df, self.TEAM, self.CUTOFF)
        assert result == pytest.approx(0.0)

    def test_empty_dataframe_gives_zero(self):
        """No games at all → QMS = 0.0 (early-return guard)."""
        # Build an explicitly empty DataFrame with the expected Torvik schema so
        # that sort_values("DATE") does not raise a KeyError on a column-less frame.
        df = pd.DataFrame(columns=["DATE", "GAME_KEY", "WINNER", "T1_RANK", "T2_RANK"])
        result = compute_qms(df, self.TEAM, self.CUTOFF)
        assert result == pytest.approx(0.0)

    # --- Mixed wins/losses with a known expected score ---

    def test_mixed_results_known_score(self):
        """
        Construct a controlled game log:
          - 2 wins vs rank 10  (top-25):  2 × 10 = 20
          - 3 wins vs rank 40  (top-50):  3 ×  7 = 21
          - 1 win  vs rank 80  (top-100): 1 ×  4 =  4
          - 2 wins vs rank 150 (below):   2 ×  1 =  2
          - 2 losses (any rank)           0 × 0  =  0
        Total = 47 pts across 10 games.
        """
        rows = (
            [self._win_row(f"2024-01-{d:02d}", 10) for d in range(1, 3)]   # 2 wins, top-25
            + [self._win_row(f"2024-01-{d:02d}", 40) for d in range(3, 6)]  # 3 wins, top-50
            + [self._win_row("2024-01-10", 80)]                              # 1 win, top-100
            + [self._win_row(f"2024-01-{d:02d}", 150) for d in range(11, 13)]  # 2 wins, below-100
            + [self._loss_row("2024-01-20", 1)]                              # loss
            + [self._loss_row("2024-01-21", 1)]                              # loss
        )
        df = _build_df(rows)
        result = compute_qms(df, self.TEAM, self.CUTOFF)
        assert result == pytest.approx(47.0)

    # --- Only last 10 games matter ---

    def test_only_last_10_games_counted(self):
        """
        Give 15 games: the first 5 are wins vs top-25 (50 pts each if counted),
        the last 10 are all losses (0 pts).  QMS must be 0 because the window
        selects only the chronologically last 10 games.
        """
        early_wins = [self._win_row(f"2024-01-{d:02d}", 1) for d in range(1, 6)]
        late_losses = [self._loss_row(f"2024-02-{d:02d}", 1) for d in range(1, 11)]
        df = _build_df(early_wins + late_losses)
        result = compute_qms(df, self.TEAM, self.CUTOFF)
        assert result == pytest.approx(0.0), (
            "Only the last 10 games should count; early wins must be ignored."
        )

    def test_last_10_games_out_of_15_partial_score(self):
        """
        15 games total:
          - First 5 games: losses (ignored — outside last-10 window)
          - Last 10 games: all wins vs rank-25 opponents → 10 × 10 = 100
        """
        early_losses = [self._loss_row(f"2024-01-{d:02d}", 1) for d in range(1, 6)]
        late_wins = [self._win_row(f"2024-02-{d:02d}", 25) for d in range(1, 11)]
        df = _build_df(early_losses + late_wins)
        result = compute_qms(df, self.TEAM, self.CUTOFF)
        assert result == pytest.approx(100.0), (
            "Only the last 10 games should count; early losses must be ignored."
        )

    # --- Games after cutoff_date are excluded ---

    def test_games_after_cutoff_are_excluded(self):
        """
        5 wins before Mar 15 + 5 wins on/after Mar 15.
        Only the pre-cutoff games feed into QMS.
        With only 5 pre-cutoff games (rank-1 wins): score = 5 × 10 = 50.
        """
        pre_cutoff = [self._win_row(f"2024-03-{d:02d}", 1) for d in range(10, 15)]  # Mar 10-14
        post_cutoff = [self._win_row(f"2024-03-{d:02d}", 1) for d in range(15, 20)]  # Mar 15-19
        df = _build_df(pre_cutoff + post_cutoff)
        result = compute_qms(df, self.TEAM, self.CUTOFF)
        assert result == pytest.approx(50.0), (
            "Games on or after the cutoff date must not count toward QMS."
        )

    def test_all_games_after_cutoff_gives_zero(self):
        """All games fall on or after the cutoff → no eligible games → QMS = 0."""
        rows = [self._win_row(f"2024-03-{d:02d}", 1) for d in range(15, 20)]
        df = _build_df(rows)
        result = compute_qms(df, self.TEAM, self.CUTOFF)
        assert result == pytest.approx(0.0)


# ── TestHistoricalSanityChecks ────────────────────────────────────────────────

class TestHistoricalSanityChecks:
    """
    Regression guards against the pre-built data/processed/features_momentum.csv.
    These tests confirm both the QMS logic and the data pipeline are intact.
    """

    @pytest.fixture(scope="class")
    def momentum_df(self):
        path = PROCESSED_DIR / "features_momentum.csv"
        if not path.exists():
            pytest.skip("features_momentum.csv not built yet — run src/features/momentum.py first.")
        return pd.read_csv(path)

    # --- QMS is non-negative for all rows ---

    def test_qms_is_non_negative(self, momentum_df):
        """QMS cannot be negative — all weights are non-negative and losses score 0."""
        negative = momentum_df[momentum_df["QMS"] < 0]
        assert negative.empty, (
            f"Found {len(negative)} rows with negative QMS:\n"
            f"{negative[['YEAR', 'TEAM', 'SEED', 'QMS']]}"
        )

    # --- Top QMS teams have SEEDs ---

    def test_high_qms_teams_have_seeds(self, momentum_df):
        """
        Teams with QMS > 80 should be tournament participants (have a SEED).
        A non-tournament team cannot legitimately accumulate very high QMS.
        """
        high_qms = momentum_df[momentum_df["QMS"] > 80]
        missing_seed = high_qms[high_qms["SEED"].isna()]
        assert missing_seed.empty, (
            f"Non-tournament teams (no SEED) have QMS > 80:\n"
            f"{missing_seed[['YEAR', 'TEAM', 'SEED', 'QMS']]}"
        )

    # --- Known historical values ---

    def test_2021_loyola_chicago_qms_is_88(self, momentum_df):
        """
        2021 Loyola Chicago (8-seed, Elite Eight Cinderella) had an exceptionally
        hot finish with high-quality wins.  Expected QMS = 88.
        """
        row = momentum_df[
            (momentum_df["TEAM"] == "Loyola Chicago") & (momentum_df["YEAR"] == 2021)
        ]
        assert len(row) == 1, "Expected exactly one row for Loyola Chicago 2021."
        qms = row.iloc[0]["QMS"]
        assert qms == pytest.approx(88.0), (
            f"Expected QMS=88 for 2021 Loyola Chicago, got {qms}."
        )

    def test_2016_villanova_qms_is_82(self, momentum_df):
        """
        2016 Villanova (2-seed, National Champion) peaked heading into March.
        Expected QMS = 82.
        """
        row = momentum_df[
            (momentum_df["TEAM"] == "Villanova") & (momentum_df["YEAR"] == 2016)
        ]
        assert len(row) == 1, "Expected exactly one row for Villanova 2016."
        qms = row.iloc[0]["QMS"]
        assert qms == pytest.approx(82.0), (
            f"Expected QMS=82 for 2016 Villanova, got {qms}."
        )

    # --- QMS ceiling: 10 games × max 10 pts = 100 ---

    def test_no_tournament_team_exceeds_100_qms(self, momentum_df):
        """
        The theoretical maximum QMS is 10 games × 10 pts/game = 100.
        No tournament team (excluding COVID 2020 and incomplete 2025 season)
        should exceed this.
        """
        tourney = momentum_df[
            momentum_df["SEED"].notna()
            & (momentum_df["YEAR"] != 2020)
            & (momentum_df["YEAR"] != 2025)
        ]
        over_100 = tourney[tourney["QMS"] > 100]
        assert over_100.empty, (
            f"Found {len(over_100)} tournament team(s) with impossible QMS > 100:\n"
            f"{over_100[['YEAR', 'TEAM', 'SEED', 'QMS']]}"
        )
