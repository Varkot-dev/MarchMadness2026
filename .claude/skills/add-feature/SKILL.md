# Skill: add-feature

**Description:** Use when the user wants to add a new feature, engineer a new column, or build a new feature module for the March Madness model.

---

## What This Skill Does

Scaffolds a new feature module under `src/features/` following the exact pattern already established in `efficiency.py`, `momentum.py`, `coaching.py`, and `candidates.py`. It also wires the new feature into `candidates.py` and adds it to `CANDIDATE_FEATURES` so SHAP selection can evaluate it automatically.

---

## Steps

### 1. Gather context before writing anything

Ask (or infer from context):
- **Feature name** — e.g., `BENCH_SCORING_PCT` (SCREAMING_SNAKE for column names)
- **Data source** — Is the raw data already in `features_coaching.csv`? Or does it need a new raw file?
- **Domain rationale** — Why would this predict tournament success? Read `SKILLS.md` for domain context.
- **Null risk** — Which years might be missing this data?

### 2. Check what's already available

```bash
python3 -c "
import pandas as pd
from config import PROCESSED_DIR
df = pd.read_csv(PROCESSED_DIR / 'features_coaching.csv')
print(list(df.columns))
"
```

If the raw data is already in `features_coaching.csv`, the feature can be computed directly in `candidates.py` without a new file.

### 3. Create the feature module

Use the template at `assets/feature_module_template.py`. Replace:
- `FEATURE_NAME` → the snake_case module name (e.g., `bench_depth`)
- `FEATURE_COL` → the SCREAMING_SNAKE column name (e.g., `BENCH_SCORING_PCT`)
- `UPSTREAM_FILE` → the CSV this module reads from (usually `features_coaching.csv` or `features_candidates.csv`)

### 4. Wire into candidates.py

Add to `CANDIDATE_FEATURES` list in `src/features/candidates.py`:
```python
CANDIDATE_FEATURES = [
    # ... existing ...
    "YOUR_FEATURE_COL",  # brief description
]
```

Add the computation inside `build_candidate_features()`:
```python
df = add_your_feature(df)   # or inline if simple
```

### 5. Validate coverage

```bash
python3 -m src.features.candidates 2>&1 | grep "YOUR_FEATURE_COL"
# Should show: YOUR_FEATURE_COL null=X.X%
# Warn if >20% null on tournament teams
```

### 6. Run SHAP selection to see if it passes

```bash
python3 -m src.models.shap_selector
# Check: mean|SHAP| > 0.01 AND sign_consistency > 0.65 or < 0.35
```

### 7. If SHAP selects it — update formula_model.py

```python
# In src/models/formula_model.py FEATURES list:
FEATURES = [
    # ... existing ...
    "YOUR_FEATURE_COL",  # mean|SHAP|=X.XXX, consistency=X.XX
]
```

Then rerun the backtest:
```bash
python3 -m src.models.formula_model
# Compare mean ESPN score vs previous (stored in data/processed/formula_backtest_summary.csv)
```

---

## Gotchas

**1. Never redefine constants from config.py.**
Import `PROCESSED_DIR`, `EXTERNAL_DIR`, `LUCK_COEFFICIENT`, `SEED_DIVERGENCE_CLIP` etc. from `config.py`. If you catch yourself writing `PROCESSED_DIR = Path("data/processed")` in a feature file, stop.

**2. The feature column must survive the matchup difference.**
When `formula_model.py` computes `feats_A - feats_B`, the result needs to make sense directionally. Ask: "if team A has a higher value than team B on this feature, does that predict A winning?" If yes, positive SHAP = selected. If the relationship is inverted or noisy, SHAP will reject it.

**3. Don't compute on all teams — only tournament teams for coverage logging.**
`features_candidates.py` logs null % filtered to `df[df["SEED"].notna() & (df["YEAR"] != 2020)]`. Non-tournament teams will have NaN for many features and that's fine — they aren't used for modeling.

**4. QMS silently degrades when Torvik is missing.**
If your feature depends on Torvik data, it will silently return 0 for years without a `torvik_{year}.csv`. Log a warning explicitly and document it in the function docstring.

**5. Team name bridging.**
If joining an external dataset by team name, use `CBB_TO_KAGGLE_NAMES` from `coaching.py` or join via Kaggle TeamID → CBB name via `_build_kaggle_to_cbb_map()` in `win_probability.py`. Raw team name strings will not match across datasets.

**6. NaN fill strategy matters.**
After joining a new feature, fill NaN with group mean (by YEAR) or 0 — but document which you chose and why. Using `fillna(0)` is wrong for features like `LUCK` (0 luck is meaningful, not "missing"). Use `fillna(df.groupby("YEAR")["FEAT"].transform("mean"))` for truly missing values.

**7. Seed Divergence clip.**
If your feature is derived from KenPom rank or seed, apply `SEED_DIVERGENCE_CLIP = (-8, 8)` if extreme outliers are possible. Check the actual value range before deciding.

**8. features_candidates.csv must be rebuilt after adding a feature.**
The SHAP selector and formula_model both prefer `features_candidates.csv`. Run `python3 -m src.features.candidates` after any change to regenerate it.
