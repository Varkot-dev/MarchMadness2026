"""
tests/test_efficiency.py — Unit tests for src/features/efficiency.py

Covers:
  - True Quality Score calculation
  - Seed Divergence calculation
  - KenPom rank → implied seed binning
  - Sanity checks against known historical values
"""

import numpy as np
import pandas as pd
import pytest

from src.features.efficiency import (
    compute_true_quality_score,
    compute_seed_divergence,
    kenpom_rank_to_implied_seed,
)


# ── kenpom_rank_to_implied_seed ───────────────────────────────────────────────

class TestKenpomRankToImpliedSeed:
    def test_rank_1_is_seed_1(self):
        assert kenpom_rank_to_implied_seed(1) == 1.0

    def test_rank_4_is_seed_1(self):
        assert kenpom_rank_to_implied_seed(4) == 1.0

    def test_rank_5_is_seed_2(self):
        assert kenpom_rank_to_implied_seed(5) == 2.0

    def test_rank_8_is_seed_2(self):
        assert kenpom_rank_to_implied_seed(8) == 2.0

    def test_rank_64_is_seed_16(self):
        assert kenpom_rank_to_implied_seed(64) == 16.0

    def test_rank_65_caps_at_16(self):
        """Teams ranked outside top 64 cap at seed 16."""
        assert kenpom_rank_to_implied_seed(65) == 16.0
        assert kenpom_rank_to_implied_seed(200) == 16.0

    def test_nan_returns_nan(self):
        assert np.isnan(kenpom_rank_to_implied_seed(np.nan))

    def test_boundary_ranks(self):
        """Every 4-rank boundary maps correctly."""
        assert kenpom_rank_to_implied_seed(9) == 3.0
        assert kenpom_rank_to_implied_seed(12) == 3.0
        assert kenpom_rank_to_implied_seed(13) == 4.0


# ── compute_true_quality_score ────────────────────────────────────────────────

class TestComputeTrueQualityScore:
    def test_positive_luck_reduces_score(self):
        """Positive luck means team was lucky — strip it out, score drops."""
        adj_em = pd.Series([30.0])
        luck = pd.Series([0.1])
        result = compute_true_quality_score(adj_em, luck, coeff=0.4)
        assert result.iloc[0] == pytest.approx(30.0 - 0.04)

    def test_negative_luck_increases_score(self):
        """Negative luck means team was unlucky — score rises after correction."""
        adj_em = pd.Series([20.0])
        luck = pd.Series([-0.05])
        result = compute_true_quality_score(adj_em, luck, coeff=0.4)
        assert result.iloc[0] == pytest.approx(20.0 + 0.02)

    def test_zero_luck_unchanged(self):
        adj_em = pd.Series([25.0])
        luck = pd.Series([0.0])
        result = compute_true_quality_score(adj_em, luck, coeff=0.4)
        assert result.iloc[0] == pytest.approx(25.0)

    def test_coeff_zero_ignores_luck(self):
        adj_em = pd.Series([15.0])
        luck = pd.Series([0.999])
        result = compute_true_quality_score(adj_em, luck, coeff=0.0)
        assert result.iloc[0] == pytest.approx(15.0)

    def test_vectorized_series(self):
        adj_em = pd.Series([10.0, 20.0, 30.0])
        luck = pd.Series([0.0, 0.1, -0.1])
        result = compute_true_quality_score(adj_em, luck, coeff=0.4)
        expected = [10.0, 20.0 - 0.04, 30.0 + 0.04]
        for i, exp in enumerate(expected):
            assert result.iloc[i] == pytest.approx(exp)


# ── compute_seed_divergence ───────────────────────────────────────────────────

class TestComputeSeedDivergence:
    def test_underseeded_positive(self):
        """Team ranked 17th nationally (implied seed 5) but given seed 4 → divergence = 4-5 = -1."""
        # Formula: actual_seed - implied_seed. Rank 17 → ceil(17/4)=5. Actual=4. 4-5=-1.
        # Negative here because the team is OVER-seeded (given seed 4, KenPom says 5).
        kenpom_rank = pd.Series([17.0])
        actual_seed = pd.Series([4.0])
        result = compute_seed_divergence(kenpom_rank, actual_seed)
        assert result.iloc[0] == pytest.approx(4.0 - 5.0)

    def test_overseeded_negative(self):
        """Team ranked 9th nationally (implied seed 3) but seeded 1 → actual-implied = 1-3 = -2."""
        # Rank 9 → ceil(9/4)=3.0 implied seed. Actual seed 1. Divergence = 1-3 = -2.
        # Negative = overseeded (committee gave them a better seed than KenPom suggests).
        kenpom_rank = pd.Series([9.0])
        actual_seed = pd.Series([1.0])
        result = compute_seed_divergence(kenpom_rank, actual_seed)
        assert result.iloc[0] == pytest.approx(1.0 - 3.0)

    def test_underseeded_true_positive(self):
        """Team ranked 1st nationally (implied seed 1) but seeded 3 → actual-implied = 3-1 = +2."""
        # Positive = underseeded (committee gave them a WORSE seed than KenPom suggests).
        # These are upset threats — the model should favor them.
        kenpom_rank = pd.Series([1.0])
        actual_seed = pd.Series([3.0])
        result = compute_seed_divergence(kenpom_rank, actual_seed)
        assert result.iloc[0] == pytest.approx(3.0 - 1.0)

    def test_perfectly_seeded_zero(self):
        """Team ranked 1st nationally, seeded 1 → divergence 0."""
        kenpom_rank = pd.Series([1.0])
        actual_seed = pd.Series([1.0])
        result = compute_seed_divergence(kenpom_rank, actual_seed)
        assert result.iloc[0] == pytest.approx(0.0)

    def test_nan_rank_returns_nan(self):
        kenpom_rank = pd.Series([np.nan])
        actual_seed = pd.Series([5.0])
        result = compute_seed_divergence(kenpom_rank, actual_seed)
        assert np.isnan(result.iloc[0])

    def test_nan_actual_seed_returns_nan(self):
        """Non-tournament teams have no seed — divergence should be NaN."""
        kenpom_rank = pd.Series([10.0])
        actual_seed = pd.Series([np.nan])
        result = compute_seed_divergence(kenpom_rank, actual_seed)
        assert np.isnan(result.iloc[0])

    def test_rank_68_boundary(self):
        assert kenpom_rank_to_implied_seed(68) == 16.0

    def test_rank_300_caps_at_16(self):
        assert kenpom_rank_to_implied_seed(300) == 16.0


# ── Sanity checks against known historical values ─────────────────────────────

class TestHistoricalSanityChecks:
    """
    Validate outputs against known tournament history.
    These are regression guards — if they fail, something broke in the pipeline.
    """

    @pytest.fixture
    def efficiency_df(self):
        """Load the actual built features from disk."""
        import pandas as pd
        from config import PROCESSED_DIR
        path = PROCESSED_DIR / "features_efficiency.csv"
        if not path.exists():
            pytest.skip("features_efficiency.csv not built yet")
        return pd.read_csv(path)

    def test_top_true_quality_scores_are_1_seeds(self, efficiency_df):
        """Top 5 TQS across all years should all be 1-seeds."""
        tourney = efficiency_df[
            efficiency_df["SEED"].notna() &
            (efficiency_df["YEAR"] != 2020) &
            (efficiency_df["YEAR"] != 2025)
        ]
        top5 = tourney.nlargest(5, "TRUE_QUALITY_SCORE")
        assert (top5["SEED"] == 1.0).all(), f"Expected all 1-seeds:\n{top5[['YEAR','TEAM','SEED','TRUE_QUALITY_SCORE']]}"

    def test_2015_kentucky_highest_tqs(self, efficiency_df):
        """2015 Kentucky (38-1) should have the highest True Quality Score."""
        tourney = efficiency_df[
            efficiency_df["SEED"].notna() &
            (efficiency_df["YEAR"] != 2020) &
            (efficiency_df["YEAR"] != 2025)
        ]
        top = tourney.nlargest(1, "TRUE_QUALITY_SCORE").iloc[0]
        assert top["YEAR"] == 2015
        assert top["TEAM"] == "Kentucky"

    def test_seed_divergence_range(self, efficiency_df):
        """Seed divergence should be bounded between -16 and +16."""
        tourney = efficiency_df[
            efficiency_df["SEED"].notna() &
            (efficiency_df["YEAR"] != 2020) &
            (efficiency_df["YEAR"] != 2025) &
            efficiency_df["SEED_DIVERGENCE"].notna()
        ]
        assert tourney["SEED_DIVERGENCE"].min() >= -16
        assert tourney["SEED_DIVERGENCE"].max() <= 16

    def test_no_null_tqs_for_tournament_teams_with_kenpom(self, efficiency_df):
        """Tournament teams that have KenPom data must have TQS."""
        tourney = efficiency_df[
            efficiency_df["SEED"].notna() &
            (efficiency_df["YEAR"] != 2020) &
            efficiency_df["KENPOM_NETRTG"].notna()
        ]
        nulls = tourney["TRUE_QUALITY_SCORE"].isna().sum()
        assert nulls == 0, f"{nulls} tournament teams with KenPom data are missing TQS"

    def test_luck_coefficient_effect(self, efficiency_df):
        """Lucky teams (positive Luck) should have TQS < KENPOM_NETRTG."""
        lucky = efficiency_df[
            efficiency_df["LUCK"] > 0.05
        ].dropna(subset=["KENPOM_NETRTG", "TRUE_QUALITY_SCORE"])
        assert (lucky["TRUE_QUALITY_SCORE"] < lucky["KENPOM_NETRTG"]).all()
