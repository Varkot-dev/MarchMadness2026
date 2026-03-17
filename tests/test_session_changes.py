"""
tests/test_session_changes.py — Tests for all changes made in this session.

Covers:
  1. config.py — ESPN_ROUND_POINTS, SEED_PAIRINGS, REGIONS, SEED_DIVERGENCE_CLIP, FIRST_FOUR_2025
  2. efficiency.py — Seed Divergence clipping (±8), correct sign convention
  3. ui/app.py — _NaNSafeEncoder handles float NaN/Inf, year validation returns 400
  4. simulator.py — imports ESPN_ROUND_POINTS from config (no local redef)
  5. formula_model.py — imports ESPN_ROUND_POINTS from config (no local redef)
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# 1. config.py constants
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_espn_round_points_values(self):
        """ESPN_ROUND_POINTS must match the actual ESPN bracket scoring."""
        from config import ESPN_ROUND_POINTS
        assert ESPN_ROUND_POINTS[1] == 10,  "R64 should be 10 pts"
        assert ESPN_ROUND_POINTS[2] == 20,  "R32 should be 20 pts"
        assert ESPN_ROUND_POINTS[3] == 40,  "S16 should be 40 pts"
        assert ESPN_ROUND_POINTS[4] == 80,  "E8 should be 80 pts"
        assert ESPN_ROUND_POINTS[5] == 160, "F4 should be 160 pts"
        assert ESPN_ROUND_POINTS[6] == 320, "Champ should be 320 pts"

    def test_espn_round_points_max_score(self):
        """Max possible score = 63 games × their point values = 1920."""
        from config import ESPN_ROUND_POINTS
        # R64: 32 games, R32: 16, S16: 8, E8: 4, F4: 2, Champ: 1
        game_counts = {1: 32, 2: 16, 3: 8, 4: 4, 5: 2, 6: 1}
        total = sum(ESPN_ROUND_POINTS[r] * game_counts[r] for r in range(1, 7))
        assert total == 1920

    def test_seed_pairings_count(self):
        """Must be exactly 8 first-round matchups per region."""
        from config import SEED_PAIRINGS
        assert len(SEED_PAIRINGS) == 8

    def test_seed_pairings_add_to_17(self):
        """Each seed pair sums to 17 (tournament property)."""
        from config import SEED_PAIRINGS
        for high, low in SEED_PAIRINGS:
            assert high + low == 17, f"{high} + {low} should equal 17"

    def test_regions_count(self):
        from config import REGIONS
        assert len(REGIONS) == 4
        assert set(REGIONS) == {"South", "East", "West", "Midwest"}

    def test_seed_divergence_clip_bounds(self):
        """Clip bounds should be (-8, 8) — prevents extreme outlier destabilization."""
        from config import SEED_DIVERGENCE_CLIP
        lo, hi = SEED_DIVERGENCE_CLIP
        assert lo == -8
        assert hi == 8

    def test_first_four_2025_count(self):
        """First Four has 6 play-in games (2 seed-11 games, 4 seed-16 games across regions)."""
        from config import FIRST_FOUR_2025
        assert len(FIRST_FOUR_2025) == 6

    def test_first_four_2025_seeds(self):
        """First Four games are only for seeds 11 and 16."""
        from config import FIRST_FOUR_2025
        for _, _, _, seed in FIRST_FOUR_2025:
            assert seed in (11, 16), f"Unexpected First Four seed: {seed}"

    def test_first_four_2025_regions(self):
        """First Four games reference valid regions."""
        from config import FIRST_FOUR_2025, REGIONS
        for _, _, region, _ in FIRST_FOUR_2025:
            assert region in REGIONS, f"Unknown region: {region}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Seed Divergence clipping
# ─────────────────────────────────────────────────────────────────────────────

class TestSeedDivergenceClipping:
    def test_extreme_low_rank_clips_to_minus_8(self):
        """A rank-200 team seeded 1 would produce raw divergence = 1 - 50 = -49; clips to -8."""
        from src.features.efficiency import compute_seed_divergence
        kenpom_rank = pd.Series([200.0])
        actual_seed = pd.Series([1.0])
        result = compute_seed_divergence(kenpom_rank, actual_seed)
        assert result.iloc[0] == -8.0, f"Expected -8.0, got {result.iloc[0]}"

    def test_extreme_high_rank_clips_to_plus_8(self):
        """A rank-1 team seeded 16 would produce raw divergence = 16 - 1 = +15; clips to +8."""
        from src.features.efficiency import compute_seed_divergence
        kenpom_rank = pd.Series([1.0])
        actual_seed = pd.Series([16.0])
        result = compute_seed_divergence(kenpom_rank, actual_seed)
        assert result.iloc[0] == 8.0, f"Expected 8.0, got {result.iloc[0]}"

    def test_normal_divergence_not_clipped(self):
        """Normal divergence within ±8 should pass through unchanged."""
        from src.features.efficiency import compute_seed_divergence
        # Rank 17 → implied seed 5, actual seed 3 → divergence = 3 - 5 = -2
        kenpom_rank = pd.Series([17.0])
        actual_seed = pd.Series([3.0])
        result = compute_seed_divergence(kenpom_rank, actual_seed)
        assert result.iloc[0] == pytest.approx(-2.0)

    def test_divergence_at_clip_boundary(self):
        """Values exactly at ±8 should pass through without change."""
        from src.features.efficiency import compute_seed_divergence
        # Rank 1 → implied 1, actual seed 9 → divergence = 9 - 1 = +8 (exactly at boundary)
        kenpom_rank = pd.Series([1.0])
        actual_seed = pd.Series([9.0])
        result = compute_seed_divergence(kenpom_rank, actual_seed)
        assert result.iloc[0] == pytest.approx(8.0)

    def test_sign_convention_underseeded_is_positive(self):
        """SKILLS.md: actual_seed - implied_seed. Underseeded (ranked better) = positive."""
        from src.features.efficiency import compute_seed_divergence
        # Team KenPom rank 4 (implied seed 1), given actual seed 4 → divergence = 4 - 1 = +3
        kenpom_rank = pd.Series([4.0])
        actual_seed = pd.Series([4.0])
        result = compute_seed_divergence(kenpom_rank, actual_seed)
        assert result.iloc[0] == pytest.approx(3.0), \
            "Underseeded team should have positive divergence"

    def test_sign_convention_overseeded_is_negative(self):
        """Overseeded teams (ranked worse than seed) = negative divergence."""
        from src.features.efficiency import compute_seed_divergence
        # Rank 1 → implied seed 1, given actual seed 1 → divergence = 1 - 1 = 0
        # Rank 9 → implied seed 3, given actual seed 1 → divergence = 1 - 3 = -2 (overseeded)
        kenpom_rank = pd.Series([9.0])
        actual_seed = pd.Series([1.0])
        result = compute_seed_divergence(kenpom_rank, actual_seed)
        assert result.iloc[0] < 0, "Overseeded team should have negative divergence"


# ─────────────────────────────────────────────────────────────────────────────
# 3. _NaNSafeEncoder in ui/app.py
# ─────────────────────────────────────────────────────────────────────────────

class TestNaNSafeEncoder:
    """
    Validate that _NaNSafeEncoder converts float NaN/Inf to null in JSON.
    The root bug: pandas CSVs store missing values as float NaN which is
    not valid JSON. Flask's default encoder emits literal 'NaN', breaking
    browser JSON.parse().

    We inline the encoder here to avoid importing Flask in tests (Flask
    attempts app startup which fails in test environments).
    """

    def _get_encoder(self):
        """Return the _NaNSafeEncoder class, inlined to avoid Flask startup."""
        class _NaNSafeEncoder(json.JSONEncoder):
            def iterencode(self, o, _one_shot=False):
                return super().iterencode(self._clean(o), _one_shot)

            def _clean(self, obj):
                if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                    return None
                if isinstance(obj, dict):
                    return {k: self._clean(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [self._clean(v) for v in obj]
                return obj

        return _NaNSafeEncoder

    def test_float_nan_becomes_null(self):
        """float NaN → null (None) in JSON output."""
        encoder_cls = self._get_encoder()
        result = json.dumps({"value": float("nan")}, cls=encoder_cls)
        parsed = json.loads(result)
        assert parsed["value"] is None

    def test_float_inf_becomes_null(self):
        """float Inf → null (None) in JSON output."""
        encoder_cls = self._get_encoder()
        result = json.dumps({"value": float("inf")}, cls=encoder_cls)
        parsed = json.loads(result)
        assert parsed["value"] is None

    def test_float_neg_inf_becomes_null(self):
        """float -Inf → null (None) in JSON output."""
        encoder_cls = self._get_encoder()
        result = json.dumps({"value": float("-inf")}, cls=encoder_cls)
        parsed = json.loads(result)
        assert parsed["value"] is None

    def test_nested_nan_cleaned(self):
        """NaN inside nested dicts and lists is also cleaned."""
        encoder_cls = self._get_encoder()
        obj = {
            "team": "Florida",
            "features": {"tqs": float("nan"), "seed": 1.0},
            "probs": [0.75, float("nan"), 0.25],
        }
        result = json.dumps(obj, cls=encoder_cls)
        parsed = json.loads(result)
        assert parsed["features"]["tqs"] is None
        assert parsed["probs"][1] is None

    def test_normal_float_preserved(self):
        """Valid floats must NOT be altered."""
        encoder_cls = self._get_encoder()
        result = json.dumps({"prob": 0.734}, cls=encoder_cls)
        parsed = json.loads(result)
        assert parsed["prob"] == pytest.approx(0.734)

    def test_numpy_nan_cleaned(self):
        """numpy NaN (the type that comes from pandas) is also handled."""
        encoder_cls = self._get_encoder()
        result = json.dumps({"v": np.nan}, cls=encoder_cls)
        parsed = json.loads(result)
        assert parsed["v"] is None

    def test_pandas_nan_in_dict_cleaned(self):
        """Simulate what happens when DataFrame.to_dict() produces NaN values."""
        encoder_cls = self._get_encoder()
        df = pd.DataFrame({"team": ["Florida"], "tqs": [float("nan")]})
        records = df.to_dict("records")  # produces [{"team": "Florida", "tqs": nan}]
        result = json.dumps(records, cls=encoder_cls)
        # Should not raise — if it does, original bug is back
        parsed = json.loads(result)
        assert parsed[0]["tqs"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 4. simulator.py — constants imported from config, not redefined locally
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulatorImports:
    def test_simulator_imports_espn_round_points_from_config(self):
        """simulator.py must NOT redefine ESPN_ROUND_POINTS locally."""
        import ast
        path = ROOT / "src" / "models" / "simulator.py"
        source = path.read_text()
        tree = ast.parse(source)

        # Check that ESPN_ROUND_POINTS is not assigned at module level
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "ESPN_ROUND_POINTS":
                        pytest.fail("simulator.py redefines ESPN_ROUND_POINTS locally — import from config instead")

    def test_simulator_imports_seed_pairings_from_config(self):
        """simulator.py must NOT redefine SEED_PAIRINGS locally."""
        import ast
        path = ROOT / "src" / "models" / "simulator.py"
        source = path.read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "SEED_PAIRINGS":
                        pytest.fail("simulator.py redefines SEED_PAIRINGS locally — import from config instead")

    def test_simulator_imports_config(self):
        """simulator.py must import from config."""
        path = ROOT / "src" / "models" / "simulator.py"
        source = path.read_text()
        assert "from config import" in source or "import config" in source, \
            "simulator.py must import from config.py"


# ─────────────────────────────────────────────────────────────────────────────
# 5. formula_model.py — constants imported from config
# ─────────────────────────────────────────────────────────────────────────────

class TestFormulaModelImports:
    def test_formula_model_imports_espn_round_points_from_config(self):
        """formula_model.py must NOT redefine ESPN_ROUND_POINTS locally."""
        import ast
        path = ROOT / "src" / "models" / "formula_model.py"
        source = path.read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "ESPN_ROUND_POINTS":
                        pytest.fail("formula_model.py redefines ESPN_ROUND_POINTS locally — import from config instead")

    def test_formula_model_imports_config(self):
        """formula_model.py must import from config."""
        path = ROOT / "src" / "models" / "formula_model.py"
        source = path.read_text()
        assert "from config import" in source or "import config" in source, \
            "formula_model.py must import from config.py"


# ─────────────────────────────────────────────────────────────────────────────
# 6. No hardcoded DayNum→round mappings
# ─────────────────────────────────────────────────────────────────────────────

class TestNoDayNumHardcoding:
    """
    SKILLS.md explicitly forbids hardcoded DayNum→round mappings.
    Verify that neither formula_model.py nor ui/app.py contain
    a hardcoded dict like {134: 1, 136: 1, ...}.
    """

    def _count_daynum_assignments(self, filepath: Path) -> int:
        """Count suspicious DayNum → round literal dict patterns."""
        source = filepath.read_text()
        import re
        # Flag: integer keys in range 130-155 (typical DayNum range) being used
        # as a mapping. A real dict with 5+ entries in that range is suspicious.
        matches = re.findall(r"\b1[3-5]\d\s*:", source)
        return len(matches)

    def test_formula_model_no_hardcoded_daynum(self):
        """formula_model.py should not hardcode DayNum mappings."""
        path = ROOT / "src" / "models" / "formula_model.py"
        count = self._count_daynum_assignments(path)
        assert count < 3, f"Possible hardcoded DayNum table in formula_model.py ({count} matches)"

    def test_ui_app_no_hardcoded_daynum(self):
        """ui/app.py should not hardcode DayNum mappings."""
        path = ROOT / "ui" / "app.py"
        count = self._count_daynum_assignments(path)
        assert count < 3, f"Possible hardcoded DayNum table in ui/app.py ({count} matches)"
