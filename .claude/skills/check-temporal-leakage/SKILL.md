# Skill: check-temporal-leakage

**Description:** Use when the user asks about data leakage, wants to verify a temporal split is correct, is reviewing model code for leakage bugs, or has surprising results that might be caused by leakage.

---

## What This Skill Does

Audits code for temporal data leakage — the most dangerous and subtle bug in this codebase. Temporal leakage means future information (data from year T) is used when predicting year T or earlier. It causes optimistically inflated CV scores that collapse on real holdout data.

---

## Checklist: Common Leakage Patterns

Run through this list when reviewing any model or feature code:

### 1. Train/test split by YEAR, not random

```python
# ✓ CORRECT — strict temporal split
prior_years = [y for y in all_years if y < test_year]   # STRICTLY less than
train = df[df["YEAR"].isin(prior_years)]
test  = df[df["YEAR"] == test_year]

# ✗ LEAKS — includes the test year in training
prior_years = [y for y in all_years if y <= test_year]  # <= is wrong

# ✗ LEAKS — random split mixes all years
train, test = train_test_split(df, test_size=0.2, random_state=42)
```

### 2. StandardScaler fit only on training data

```python
# ✓ CORRECT
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit+transform on train
X_test_scaled  = scaler.transform(X_test)         # transform only on test

# ✗ LEAKS — scaler sees test distribution
X_all_scaled = scaler.fit_transform(X_all)        # fit on everything
```

### 3. Coach premium uses expanding window

In `coaching.py`, `compute_coach_premiums()` computes premium with an expanding window:
```python
# For coach's first appearance: premium = 0
# For subsequent years: premium = (prior wins) - (prior expected)
# This is CORRECT — uses only past data at each year's training time
```

Do NOT accidentally add the current year's wins to the premium calculation.

### 4. DayNum→round mapping uses only that year's games

```python
# ✓ CORRECT — _build_daynum_to_round uses only year's own results
from src.evaluation.backtest import _build_daynum_to_round
day_map = _build_daynum_to_round(results_df[results_df["Season"] == year])

# ✗ LEAKS — would leak game counts from future years
day_map = _build_daynum_to_round(results_df)  # all years at once
```

### 5. SHAP selection must not use holdout years

In `shap_selector.py`, CV_YEARS must not include holdout years (2022–2024) in the training set when selecting features. But the SHAP selector's CV uses those years as TEST folds — that's fine and correct.

```python
# ✓ CORRECT — 2022 test fold only trains on 2013-2021
for year in CV_YEARS:   # [2016, ..., 2024]
    prior = [y for y in all_years if y < year]   # strictly prior

# ✗ LEAKS — if SHAP selection was done on all data including holdouts,
#   then FEATURES selection would be biased toward what works 2022-2024
```

### 6. Feature engineering: no future data in rolling averages

If adding any rolling/cumulative feature (e.g., "last 3 years win rate"):
```python
# ✓ CORRECT — only prior years
prior_seasons = df[df["YEAR"] < current_year]

# ✗ LEAKS — includes current year's games in "prior" signal
all_seasons = df[df["YEAR"] <= current_year]
```

### 7. 2020 is excluded, not leaked

2020 had no tournament (COVID). Including it in train/test loops silently produces a fold with 0 games, which skews metrics. It should always be excluded:
```python
# ✓ Exclude 2020 everywhere
df = df[df["YEAR"] != 2020]
```

---

## How to Audit a File

When reading a model or feature file, grep for these patterns:

```bash
# Check for random splits (leakage red flag)
grep -n "train_test_split\|random_state.*split\|sample(" src/models/formula_model.py

# Check for <= in year comparisons (common off-by-one leakage)
grep -n "< *test_year\|<= *test_year\|YEAR.*<=" src/models/formula_model.py

# Check scaler usage
grep -n "fit_transform\|\.fit(" src/models/formula_model.py

# Check 2020 exclusion
grep -n "2020" src/models/formula_model.py
```

---

## Diagnosing Suspected Leakage

If CV accuracy is suspiciously high (>85% on tournament games), run:

```bash
python3 -c "
from src.models.formula_model import load_matchup_data, run_temporal_cv, TRAIN_YEARS
import pandas as pd

df = load_matchup_data()
train = df[df['YEAR'].isin([y for y in TRAIN_YEARS if y != 2020])]

# Normal temporal CV (should be ~78% accuracy)
cv, c = run_temporal_cv(train)
print('Temporal CV (strict):')
print(cv[['year','accuracy','log_loss']].to_string())
print(f'Mean acc: {cv.accuracy.mean():.4f}')

# Check: does removing strict ordering change results?
# If adding test year to training improves by >2%, leakage is likely
"
```

**Benchmark:** Temporal CV should produce ~75–80% accuracy. Anything above 85% with only 600 training games is a red flag.

---

## Gotchas

**1. LOSO-CV vs temporal CV give different accuracy — that's expected.**
- LOSO-CV: trains on all OTHER years including future ones → optimistic (~80%)
- Temporal CV: trains on PRIOR years only → realistic (~78%)
The difference (~2%) is the "time cost" of not seeing future data. Small gap = good model generalization.

**2. Coach premium is an expanding window, not a rolling window.**
It uses ALL prior years, not just recent N. This is correct: Tom Izzo's entire career is the signal, not just the last 3 years. Don't "fix" this to a rolling window without testing.

**3. 2025 prediction is NOT evaluated by temporal CV.**
The 2025 bracket is a live prediction trained on all 2013–2024 data. There's no test fold for 2025. The only validation is the holdout backtest on 2022–2024.

**4. The FEATURES list in formula_model.py was selected by SHAP on 2016–2024 folds.**
This means the feature selection process DID see 2022–2024 data (as test folds for SHAP). This is valid — SHAP selection is unsupervised feature filtering, not supervised model training. But it means the 7 features are selected knowing which features are stable across 2016–2024, which could be mildly optimistic.
