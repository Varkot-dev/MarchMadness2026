"""
tests/test_coaching.py — Unit tests for src/features/coaching.py

Covers:
  - expected_wins() lookup
  - compute_coach_premiums() expanding window + no-leakage
  - load_kaggle_coach_lookup() mid-season change handling
  - build_coaching_features() end-to-end sanity (no duplicates, known coaches)
"""

import pandas as pd
import pytest

from src.features.coaching import (
    expected_wins,
    compute_coach_premiums,
    load_kaggle_coach_lookup,
    EXPECTED_WINS_BY_SEED,
)


# ── expected_wins ──────────────────────────────────────────────────────────────

class TestExpectedWins:
    def test_seed_1_highest(self):
        assert expected_wins(1) == EXPECTED_WINS_BY_SEED[1]

    def test_seed_16_lowest(self):
        assert expected_wins(16) == EXPECTED_WINS_BY_SEED[16]

    def test_seed_1_greater_than_seed_2(self):
        assert expected_wins(1) > expected_wins(2)

    def test_all_seeds_positive(self):
        for seed in range(1, 17):
            assert expected_wins(seed) > 0

    def test_nan_returns_default(self):
        import math
        result = expected_wins(float("nan"))
        assert result == 0.5

    def test_unknown_seed_returns_default(self):
        assert expected_wins(17) == 0.5


# ── compute_coach_premiums ─────────────────────────────────────────────────────

class TestComputeCoachPremiums:
    def _make_input(self, records: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(records)

    def test_first_appearance_is_zero(self):
        """Coach with no prior history gets premium = 0.0."""
        data = self._make_input([
            {"YEAR": 2015, "COACH": "Coach A", "TOURNEY_W": 3, "SEED": 2.0},
        ])
        result = compute_coach_premiums(data)
        row = result[result["YEAR"] == 2015].iloc[0]
        assert row["COACH_PREMIUM"] == 0.0

    def test_second_year_reflects_first_year_performance(self):
        """Year 2 premium = year 1 wins - year 1 expected wins."""
        data = self._make_input([
            {"YEAR": 2015, "COACH": "Coach A", "TOURNEY_W": 6, "SEED": 1.0},
            {"YEAR": 2016, "COACH": "Coach A", "TOURNEY_W": 0, "SEED": 1.0},
        ])
        result = compute_coach_premiums(data).sort_values("YEAR")
        year2 = result[result["YEAR"] == 2016].iloc[0]
        expected = 6.0 - expected_wins(1.0)
        assert year2["COACH_PREMIUM"] == pytest.approx(expected, rel=1e-3)

    def test_no_future_leakage(self):
        """Year N premium must not include year N wins."""
        data = self._make_input([
            {"YEAR": 2015, "COACH": "Coach B", "TOURNEY_W": 6, "SEED": 1.0},
            {"YEAR": 2016, "COACH": "Coach B", "TOURNEY_W": 6, "SEED": 1.0},
            {"YEAR": 2017, "COACH": "Coach B", "TOURNEY_W": 0, "SEED": 1.0},
        ])
        result = compute_coach_premiums(data).sort_values("YEAR")
        # Year 2017 should only include 2015+2016 data
        year_2017 = result[result["YEAR"] == 2017].iloc[0]
        expected = (6.0 + 6.0) - (expected_wins(1.0) + expected_wins(1.0))
        assert year_2017["COACH_PREMIUM"] == pytest.approx(expected, rel=1e-3)

    def test_underperformer_gets_negative_premium(self):
        """Coach winning fewer games than seed expects gets negative premium."""
        data = self._make_input([
            # Seed 1 expects 3.2 wins; coach won 0 (first round exit)
            {"YEAR": 2015, "COACH": "Coach C", "TOURNEY_W": 0, "SEED": 1.0},
            {"YEAR": 2016, "COACH": "Coach C", "TOURNEY_W": 0, "SEED": 1.0},
        ])
        result = compute_coach_premiums(data).sort_values("YEAR")
        year_2016 = result[result["YEAR"] == 2016].iloc[0]
        assert year_2016["COACH_PREMIUM"] < 0

    def test_multiple_coaches_independent(self):
        """Two coaches do not affect each other's premiums."""
        data = self._make_input([
            {"YEAR": 2015, "COACH": "Coach X", "TOURNEY_W": 6, "SEED": 1.0},
            {"YEAR": 2016, "COACH": "Coach X", "TOURNEY_W": 0, "SEED": 1.0},
            {"YEAR": 2015, "COACH": "Coach Y", "TOURNEY_W": 0, "SEED": 1.0},
            {"YEAR": 2016, "COACH": "Coach Y", "TOURNEY_W": 0, "SEED": 1.0},
        ])
        result = compute_coach_premiums(data)
        x_2016 = result[(result["COACH"] == "Coach X") & (result["YEAR"] == 2016)].iloc[0]
        y_2016 = result[(result["COACH"] == "Coach Y") & (result["YEAR"] == 2016)].iloc[0]
        # Coach X had a great 2015; Coach Y did not — premiums must differ
        assert x_2016["COACH_PREMIUM"] > y_2016["COACH_PREMIUM"]

    def test_output_columns(self):
        data = self._make_input([
            {"YEAR": 2015, "COACH": "Coach A", "TOURNEY_W": 2, "SEED": 3.0},
        ])
        result = compute_coach_premiums(data)
        assert set(result.columns) >= {"YEAR", "COACH", "COACH_PREMIUM"}


# ── load_kaggle_coach_lookup ───────────────────────────────────────────────────

class TestLoadKaggleCoachLookup:
    @pytest.fixture
    def lookup(self):
        from config import EXTERNAL_DIR
        coaches_path = EXTERNAL_DIR / "kaggle" / "MTeamCoaches.csv"
        teams_path = EXTERNAL_DIR / "kaggle" / "MTeams.csv"
        if not coaches_path.exists() or not teams_path.exists():
            pytest.skip("Kaggle source files not present")
        return load_kaggle_coach_lookup(coaches_path, teams_path)

    def test_expected_columns(self, lookup):
        assert set(lookup.columns) == {"YEAR", "KAGGLE_TEAM", "COACH"}

    def test_no_duplicate_team_years(self, lookup):
        """Each team should appear at most once per year."""
        dups = lookup.groupby(["YEAR", "KAGGLE_TEAM"]).size()
        assert (dups == 1).all(), f"Duplicate team-years found:\n{dups[dups > 1]}"

    def test_coach_names_are_title_case(self, lookup):
        """Coach names should be Title Case (e.g. 'Tom Izzo', not 'tom_izzo')."""
        sample = lookup["COACH"].dropna().head(50)
        for name in sample:
            assert name == name.title(), f"Not title case: {name!r}"

    def test_known_coach_team_pair(self, lookup):
        """Tom Izzo should map to Michigan St. every year."""
        izzo = lookup[lookup["COACH"] == "Tom Izzo"]
        assert len(izzo) > 0, "Tom Izzo not found in lookup"
        assert (izzo["KAGGLE_TEAM"] == "Michigan St").all(), \
            f"Unexpected teams for Tom Izzo:\n{izzo}"


# ── Historical sanity checks ───────────────────────────────────────────────────

class TestHistoricalSanityChecks:
    @pytest.fixture
    def coaching_df(self):
        from config import PROCESSED_DIR
        path = PROCESSED_DIR / "features_coaching.csv"
        if not path.exists():
            pytest.skip("features_coaching.csv not built yet")
        return pd.read_csv(path)

    def test_no_coach_assigned_to_multiple_tourney_teams_same_year(self, coaching_df):
        """No coach should appear on more than one tournament team in the same year."""
        tourney = coaching_df[coaching_df["SEED"].notna() & coaching_df["COACH"].notna()]
        dups = tourney.groupby(["YEAR", "COACH"])["TEAM"].nunique()
        bad = dups[dups > 1]
        assert len(bad) == 0, f"Coaches matched to multiple teams:\n{bad}"

    def test_tom_izzo_only_michigan_state(self, coaching_df):
        """Tom Izzo should only appear on Michigan St. rows."""
        izzo_rows = coaching_df[coaching_df["COACH"] == "Tom Izzo"]
        assert len(izzo_rows) > 0
        assert (izzo_rows["TEAM"] == "Michigan St.").all(), \
            f"Unexpected teams:\n{izzo_rows[['YEAR','TEAM']].to_string()}"

    def test_coach_premium_increases_after_good_run(self, coaching_df):
        """Tom Izzo's premium in 2016 should be higher than 2013 (his 2013-2015 runs were strong)."""
        izzo = coaching_df[coaching_df["COACH"] == "Tom Izzo"].set_index("YEAR")
        premium_2013 = izzo.loc[2013, "COACH_PREMIUM"]
        premium_2016 = izzo.loc[2016, "COACH_PREMIUM"]
        assert premium_2016 > premium_2013

    def test_coach_premium_range_is_sane(self, coaching_df):
        """Premiums should be bounded — no coach is 30 wins above expectation."""
        tourney = coaching_df[coaching_df["SEED"].notna() & (coaching_df["YEAR"] != 2020)]
        assert tourney["COACH_PREMIUM"].max() < 30
        assert tourney["COACH_PREMIUM"].min() > -30

    def test_first_year_coach_premium_is_zero(self, coaching_df):
        """Every coach's first tournament appearance should have premium = 0.0."""
        has_coach = coaching_df[coaching_df["COACH"].notna() & coaching_df["SEED"].notna()]
        first_years = has_coach.sort_values("YEAR").groupby("COACH").first().reset_index()
        non_zero = first_years[first_years["COACH_PREMIUM"] != 0.0]
        assert len(non_zero) == 0, \
            f"Coaches with non-zero first-year premium:\n{non_zero[['COACH','YEAR','COACH_PREMIUM']]}"

    def test_coach_premium_not_null_for_tournament_teams_with_coach(self, coaching_df):
        """Any tournament team with a matched coach must have a non-null premium."""
        tourney = coaching_df[coaching_df["SEED"].notna() & coaching_df["COACH"].notna()]
        nulls = tourney["COACH_PREMIUM"].isna().sum()
        assert nulls == 0, f"{nulls} rows have null COACH_PREMIUM despite having a coach"
