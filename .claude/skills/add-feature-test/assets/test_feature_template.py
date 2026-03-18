"""
test_FEATURE_NAME.py — Tests for src/features/FEATURE_NAME.py

Covers:
  - Unit tests: pure function behavior on controlled inputs
  - Edge cases: NaN handling, zero division, boundary values
  - Integration tests: statistical validation on real CSV data
  - Regression guards: known historical facts that must always hold
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import PROCESSED_DIR
from src.features.FEATURE_NAME import (
    compute_FEATURE_COL,
    build_FEATURE_NAME_features,
    # Import any other pure functions to test
)


# ── Unit tests — pure function behavior ──────────────────────────────────────

class TestComputeFEATURE_COL:
    """Unit tests for the core computation function."""

    def test_basic_case(self):
        """[Describe what this test proves about the basic case.]"""
        col_a = pd.Series([120.0])
        col_b = pd.Series([90.0])
        result = compute_FEATURE_COL(col_a, col_b)
        assert result.iloc[0] == pytest.approx(120.0 / 90.0)

    def test_nan_propagates(self):
        """NaN input must return NaN, not 0 or an error."""
        col_a = pd.Series([np.nan])
        col_b = pd.Series([90.0])
        result = compute_FEATURE_COL(col_a, col_b)
        assert pd.isna(result.iloc[0])

    def test_nan_in_denominator(self):
        """NaN denominator must return NaN."""
        col_a = pd.Series([120.0])
        col_b = pd.Series([np.nan])
        result = compute_FEATURE_COL(col_a, col_b)
        assert pd.isna(result.iloc[0])

    def test_zero_denominator(self):
        """Zero denominator must return NaN, not inf or ZeroDivisionError."""
        col_a = pd.Series([120.0])
        col_b = pd.Series([0.0])
        result = compute_FEATURE_COL(col_a, col_b)
        assert pd.isna(result.iloc[0]) or not np.isinf(result.iloc[0])

    def test_vectorized(self):
        """Function works on multi-row Series (not just scalars)."""
        col_a = pd.Series([120.0, 110.0, 100.0])
        col_b = pd.Series([90.0,  100.0, 110.0])
        result = compute_FEATURE_COL(col_a, col_b)
        assert len(result) == 3

    def test_negative_values(self):
        """[Describe expected behavior with negative inputs if applicable.]"""
        col_a = pd.Series([-5.0])
        col_b = pd.Series([10.0])
        result = compute_FEATURE_COL(col_a, col_b)
        assert result.iloc[0] == pytest.approx(-0.5)

    # Add domain-specific unit tests here:
    # def test_SPECIFIC_CASE(self):
    #     """[What tournament-specific behavior does this verify?]"""
    #     ...


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_all_nan_series(self):
        """All-NaN input should return all-NaN output, not raise."""
        col_a = pd.Series([np.nan, np.nan])
        col_b = pd.Series([np.nan, np.nan])
        result = compute_FEATURE_COL(col_a, col_b)
        assert result.isna().all()

    def test_empty_series(self):
        """Empty Series input should return empty output."""
        result = compute_FEATURE_COL(pd.Series([], dtype=float), pd.Series([], dtype=float))
        assert len(result) == 0

    def test_boundary_values(self):
        """[Test boundary values specific to this feature.]"""
        # Example: test that result is clipped to expected range
        # col_a = pd.Series([1000.0])  # Extreme outlier
        # col_b = pd.Series([1.0])
        # result = compute_FEATURE_COL(col_a, col_b)
        # assert result.iloc[0] <= EXPECTED_MAX  # Replace with actual bound
        pass


# ── Integration tests — real data validation ─────────────────────────────────

@pytest.fixture
def features_df():
    """Load features_candidates.csv if it exists, skip otherwise."""
    path = PROCESSED_DIR / "features_candidates.csv"
    if not path.exists():
        pytest.skip(
            "features_candidates.csv not found. "
            "Run: python3 -m src.features.candidates"
        )
    return pd.read_csv(path)


@pytest.fixture
def tourney_df(features_df):
    """Tournament teams only (SEED not null, excl. 2020)."""
    return features_df[
        features_df["SEED"].notna() &
        (features_df["YEAR"] != 2020)
    ].copy()


class TestIntegration:

    def test_feature_col_exists(self, tourney_df):
        """FEATURE_COL column must be present in output CSV."""
        assert "FEATURE_COL" in tourney_df.columns

    def test_null_rate_acceptable(self, tourney_df):
        """FEATURE_COL should have <20% null in tournament teams."""
        null_pct = tourney_df["FEATURE_COL"].isna().mean() * 100
        assert null_pct < 20, (
            f"FEATURE_COL has {null_pct:.1f}% null in tournament teams — "
            "check data source coverage"
        )

    def test_value_range(self, tourney_df):
        """FEATURE_COL values should fall within plausible range."""
        vals = tourney_df["FEATURE_COL"].dropna()
        # Replace EXPECTED_MIN and EXPECTED_MAX with domain-appropriate bounds
        EXPECTED_MIN = 0.0    # TODO: set based on domain knowledge
        EXPECTED_MAX = 100.0  # TODO: set based on domain knowledge
        assert vals.min() >= EXPECTED_MIN, f"Minimum {vals.min()} below expected {EXPECTED_MIN}"
        assert vals.max() <= EXPECTED_MAX, f"Maximum {vals.max()} above expected {EXPECTED_MAX}"

    def test_year_coverage(self, tourney_df):
        """FEATURE_COL should have data for 2022, 2023, 2024 (holdout years)."""
        for year in [2022, 2023, 2024]:
            yr_df = tourney_df[tourney_df["YEAR"] == year]
            non_null = yr_df["FEATURE_COL"].notna().sum()
            assert non_null > 30, (
                f"Only {non_null} non-null FEATURE_COL values in {year} — "
                "model won't have enough signal for holdout year"
            )


# ── Regression guards — known historical facts ───────────────────────────────

class TestHistoricalFacts:
    """
    Tests that assert known tournament history must remain stable.
    If these fail after a data change, something broke in the pipeline.
    Document the year in each test so future developers know the source.
    """

    def test_uconn_2024_top_quartile(self, tourney_df):
        """
        UConn 2024 won the championship — should rank in top 25% of
        FEATURE_COL for their year (adjust if feature doesn't predict champions).
        """
        yr_teams = tourney_df[tourney_df["YEAR"] == 2024]
        uconn = yr_teams[yr_teams["TEAM"] == "Connecticut"]
        if uconn.empty:
            pytest.skip("Connecticut 2024 not found in features — check team name mapping")
        rank_pct = yr_teams["FEATURE_COL"].rank(pct=True, ascending=True)
        uconn_pct = rank_pct[uconn.index[0]]
        assert uconn_pct >= 0.50, (
            f"UConn 2024 ranks at {uconn_pct:.0%} percentile on FEATURE_COL — "
            "expected top 50% for a champion"
        )

    # Add additional regression guards here based on your feature's domain:
    # def test_KNOWN_FACT(self, tourney_df):
    #     """[Year and source of this fact.]"""
    #     ...
