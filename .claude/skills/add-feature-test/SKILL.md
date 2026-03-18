# Skill: add-feature-test

**Description:** Use when the user wants to write tests for a feature module, add unit tests, add integration tests, or verify a feature function works correctly.

---

## What This Skill Does

Scaffolds a test file under `tests/` following the exact pattern from `test_efficiency.py` — unit tests for pure functions, integration tests that load real CSVs, and regression guards against known historical facts.

---

## Steps

### 1. Identify what to test

For any feature module, there are three layers:

| Layer | What to test | Example |
|---|---|---|
| **Unit** | Pure function behavior on controlled inputs | `compute_true_quality_score(pd.Series([30.0]), pd.Series([0.1]))` |
| **Edge cases** | NaN handling, zero division, boundary values | `compute_FEAT(pd.Series([np.nan]))` → NaN (not 0, not error) |
| **Integration** | Load real CSV, validate statistical ranges | TQS for tournament teams has plausible distribution |
| **Regression guard** | Assert known historical facts | 2015 Kentucky should have the highest TQS |

### 2. Create the test file

Use the template at `assets/test_feature_template.py`. Name it:
- `tests/test_FEATURE_NAME.py`

### 3. Run tests

```bash
pytest tests/test_FEATURE_NAME.py -v
# Or run all:
pytest tests/ -v
```

---

## Gotchas

**1. NaN must propagate, not silently become 0.**
The most common bug in feature engineering is replacing NaN with 0 where NaN is the correct answer. Tests must assert that NaN input → NaN output, not `0.0`.

```python
# Correct test:
result = compute_tqs(adj_em=pd.Series([np.nan]), luck=pd.Series([0.1]))
assert pd.isna(result.iloc[0])

# This would miss the bug:
result = compute_tqs(adj_em=pd.Series([np.nan]), luck=pd.Series([0.1]))
assert result.iloc[0] == 0.0  # BAD — 0 is not the right answer
```

**2. Test on Series, not scalars.**
All feature functions operate on `pd.Series`. If you test `compute_tqs(30.0, 0.1)`, it might pass even though the function breaks on a 1000-row DataFrame. Always pass `pd.Series([value])`.

**3. Integration tests must be conditional on CSV existing.**
CSVs in `data/processed/` may not be built yet in a clean checkout. Use `pytest.importorskip` or `pytest.fixture` with a `skipif`:

```python
@pytest.fixture
def features_df():
    path = PROCESSED_DIR / "features_candidates.csv"
    if not path.exists():
        pytest.skip("features_candidates.csv not built — run python3 -m src.features.candidates")
    return pd.read_csv(path)
```

**4. Regression guards depend on known tournament history — document the year.**
```python
# Good — year is documented
def test_uconn_2024_high_tqs(features_df):
    """UConn 2024 was dominant — should rank top 5 in TQS."""
    uconn = features_df[(features_df["TEAM"] == "Connecticut") & (features_df["YEAR"] == 2024)]
    year_teams = features_df[features_df["YEAR"] == 2024]
    rank = year_teams["TRUE_QUALITY_SCORE"].rank(ascending=False)
    assert rank[uconn.index[0]] <= 5

# Bad — no year context
def test_good_team_has_high_tqs(features_df):
    assert features_df["TRUE_QUALITY_SCORE"].max() > 30  # what does this prove?
```

**5. SEED_DIVERGENCE range check: clip matters.**
Raw divergence can reach ±49. The clipped column should always be in [-8, 8]. Test this explicitly:

```python
def test_seed_divergence_clipped(features_df):
    assert features_df["SEED_DIVERGENCE"].min() >= -8
    assert features_df["SEED_DIVERGENCE"].max() <= 8
```

**6. Vectorization: test with multi-row Series.**
Single-element Series can mask broadcasting bugs. Always include a multi-row test:

```python
def test_eff_ratio_vectorized():
    adjoe = pd.Series([120.0, 110.0, 100.0])
    adjde = pd.Series([90.0,  100.0, 110.0])
    result = compute_eff_ratio(adjoe, adjde)
    assert len(result) == 3
    assert result.iloc[0] == pytest.approx(120.0 / 90.0)
```
