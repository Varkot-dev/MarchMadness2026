# Skill: model-experiment

**Description:** Use when the user wants to try a new model, test a different feature set, compare algorithms, or run a one-off experiment without modifying the production formula_model.py.

---

## What This Skill Does

Scaffolds a self-contained experiment script under `src/models/` that reuses the existing temporal CV harness, loads data correctly, and produces a results comparison vs the current baseline. Keeps experiments isolated so the production model isn't broken mid-experiment.

---

## Steps

### 1. Name and scope the experiment

Before writing code, clarify:
- **What's changing?** Features, algorithm, hyperparameters, target?
- **What's the hypothesis?** "Adding BARTHAG should improve CV log loss."
- **Success criterion?** Mean ESPN > 1077 OR mean CV accuracy > 0.780.

### 2. Create the experiment file

Use the template at `assets/experiment_template.py`. Name it descriptively:
- `src/models/exp_add_barthag.py`
- `src/models/exp_xgboost_classifier.py`
- `src/models/exp_rounds_won_target.py`

### 3. Run and compare

```bash
python3 src/models/exp_YOUR_EXPERIMENT.py
# It will print a comparison table vs baseline automatically
```

### 4. Decide and act

| Outcome | Action |
|---|---|
| Improves both CV + holdout ESPN | Update `formula_model.py` FEATURES, commit |
| Improves CV but hurts ESPN | Investigate — likely hurts champion picks; try hybrid |
| Hurts both | Discard experiment file (don't commit) |
| Mixed (better 2024, worse 2022) | More holdout years needed; tentatively adopt if improvement is large |

### 5. If adopting: update formula_model.py

```python
# In src/models/formula_model.py
FEATURES = [
    # Updated based on exp_YOUR_EXPERIMENT.py results
    # Add: mean|SHAP|=X.XXXX, consistency=X.XX comment
]
```

Then: `python3 -m src.models.formula_model` for final validation.

---

## Key APIs to Reuse

**Load matchup data (preferred — uses features_candidates.csv):**
```python
from src.models.formula_model import load_matchup_data
df = load_matchup_data()
# df columns: FEATURES diff columns + LABEL + YEAR
```

**Run temporal CV:**
```python
from src.models.formula_model import run_temporal_cv
cv_results, best_c = run_temporal_cv(df)
```

**Fit model:**
```python
from src.models.formula_model import fit_model
model, scaler, raw_coefs = fit_model(df, c=best_c)
```

**Simulate holdout bracket:**
```python
from src.models.formula_model import simulate_bracket
result = simulate_bracket(year=2024, model=model, scaler=scaler)
print(result["espn_score"], result["champion"])
```

**Build win probability matrix:**
```python
from src.models.formula_model import build_win_prob_matrix_from_formula
matrix = build_win_prob_matrix_from_formula(2025, model, scaler)
```

---

## Gotchas

**1. Never import from formula_model.py and then modify FEATURES in the same run.**
`load_matchup_data()` in formula_model reads from the FEATURES list at module load time. If you change FEATURES mid-script, the data won't update. Always set FEATURES before loading data, or pass a custom features list explicitly.

**2. Symmetric matchup rows: always double them.**
For every game (A beats B), you need TWO rows: `(A-B, label=1)` and `(B-A, label=0)`. If you forget the second row, the model learns a directional bias (always predicts team listed first wins).

**3. XGBoost classifier vs regressor are different models for different targets.**
- `XGBClassifier` → binary label (A beats B? 1/0) → use for game outcome prediction
- `XGBRegressor` → continuous target (rounds won, 0–6) → use for team-level SHAP selection
Don't mix them up. The SHAP selector uses regressor. The main model uses LR classifier.

**4. Log loss ≠ ESPN score — measure both.**
Always compute both CV log_loss AND holdout ESPN score. Log loss can improve while ESPN drops (if the model gets calibrated but picks the wrong champion). The competition cares only about ESPN.

**5. StandardScaler must be fit on training data only.**
In temporal CV: `scaler.fit_transform(X_train)`, then `scaler.transform(X_test)`. Never `fit_transform(X_test)` — that leaks test distribution into the scaler.

**6. Don't commit experiment files.**
Experiment scripts are scratch work. If the experiment is adopted, its insight goes into `formula_model.py`. The experiment file itself should be deleted or kept in a `/experiments/` folder if you want a record — not under `src/models/`.

**7. The C grid search runs temporal CV twice in formula_model.py.**
`run_full_pipeline()` calls `run_temporal_cv()` which internally loops over the c_grid. This means ~6 × (n_years) model fits. On 8 folds × 6 C values = 48 fits. That's fast for LR but slow if you swap in XGBoost. Consider reducing c_grid = [0.1, 1.0] for fast iteration.
