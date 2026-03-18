# Skill: run-backtest

**Description:** Use when the user wants to validate the model, run cross-validation, backtest on holdout years, compare ESPN scores, or check if a model change actually improved performance.

---

## What This Skill Does

Guides running the correct validation procedure for this project. There are two distinct validation concepts and it's easy to confuse them:

| Concept | What it measures | Entry point |
|---|---|---|
| **Temporal CV** | Per-fold accuracy/log-loss on training years | `formula_model.py` (run_temporal_cv) |
| **Holdout backtest** | Actual ESPN bracket score on unseen years | `formula_model.py` (main) |
| **Layer 1 LOSO-CV** | LR vs XGBoost game-by-game accuracy | `win_probability.py` |

**The number you actually care about for competition: holdout ESPN score (2022–2024 mean).**
CV log loss is a proxy that doesn't directly correlate with bracket score.

---

## Quick Commands

**Run full holdout backtest (formula model):**
```bash
python3 -m src.models.formula_model 2>&1 | grep -E "(Year|Predicted|Actual|ESPN score|Mean ESPN|Champion acc)"
```

**Run temporal CV only (fast, no bracket simulation):**
```bash
python3 -c "
from src.models.formula_model import load_matchup_data, run_temporal_cv, TRAIN_YEARS
df = load_matchup_data()
train = df[df['YEAR'].isin([y for y in TRAIN_YEARS if y != 2020])]
cv, best_c = run_temporal_cv(train)
print(cv.to_string())
print(f'Mean acc: {cv.accuracy.mean():.4f}, Mean LL: {cv.log_loss.mean():.4f}, Best C: {best_c}')
"
```

**Compare two model versions:**
```bash
# Before change: save current results
python3 -m src.models.formula_model 2>&1 | tee /tmp/before.txt

# After change: run again
python3 -m src.models.formula_model 2>&1 | tee /tmp/after.txt

# Compare ESPN scores
grep "ESPN score\|Mean ESPN\|Champion acc" /tmp/before.txt /tmp/after.txt
```

**Run Layer 1 LOSO-CV (win_probability model):**
```bash
python3 -m src.models.win_probability 2>&1 | grep -E "(LOSO|accuracy|log_loss|Brier)"
```

---

## How to Interpret Results

### Temporal CV output
```
year  n_train  n_test  accuracy  log_loss  C
2016      164      56    0.7321    0.5102  1.0
2017      217      55    0.7818    0.4812  1.0
...
Mean accuracy:  0.7804
Mean log loss:  0.4514
```

- **accuracy > 0.75** is strong for tournament prediction (base rate ≈ 0.67 for 1-seeds)
- **log_loss < 0.50** means calibrated probabilities (random = 0.693)
- If accuracy improves but log_loss doesn't → model is guessing confidently wrong somewhere

### Holdout ESPN output
```
Year 2022:  ESPN score = 640    (champion wrong: Houston predicted, Kansas won)
Year 2023:  ESPN score = 660    (champion wrong: Houston predicted, UConn won)
Year 2024:  ESPN score = 1190   (champion correct: UConn ✓)
Mean: 830 / 1920
```

- **Max possible ESPN score: 1920**
- **Random bracket baseline: ~550–650**
- **Top 10% ESPN brackets typically score 1200+**
- **Champion pick = ~320 pts (1/6 of max)** — getting the champion right matters enormously

### What "better" looks like
- Previous 2-feature model baseline: mean ESPN ≈ 1077, champion 2/3
- Any change that drops mean ESPN below 900 is a regression
- A change that improves CV log loss but drops holdout ESPN is NOT an improvement

---

## Debugging a Bad Backtest

If ESPN scores are suddenly worse, check these in order:

**1. Did the FEATURES list change?**
```bash
git diff src/models/formula_model.py | grep "FEATURES"
```

**2. Are features_candidates.csv rebuilt after the last feature change?**
```bash
python3 -m src.features.candidates  # Rebuilds if needed
python3 -m src.models.formula_model
```

**3. Is temporal leakage happening?**
```python
# In run_temporal_cv, verify:
prior = [y for y in train_years if y < year]  # STRICTLY less than
# Not: if y <= year   (that leaks the test year's own data)
```

**4. Is the DayNum→round mapping correct?**
```python
# backtest.py uses _build_daynum_to_round() dynamically — never hardcode
# Check that it's being imported, not re-implemented inline
from src.evaluation.backtest import _build_daynum_to_round
```

**5. Are 2022–2024 seed/team names in Kaggle files?**
```bash
python3 -c "
import pandas as pd
seeds = pd.read_csv('data/external/kaggle/MNCAATourneySeeds.csv')
print(seeds[seeds.Season.isin([2022,2023,2024])].Season.value_counts())
"
# If 2022-2024 are missing, results will be empty / wrong
```

---

## Gotchas

**1. Only 3 holdout years = high variance.**
2022 (580 pts) vs 2024 (1460 pts) is a 2.5× difference from the same model. The mean (1077) is not stable. Don't declare victory from one improved year.

**2. CV log loss and ESPN score are weakly correlated.**
A model can have better CV log loss (more calibrated) but worse ESPN score (picks wrong champion). ESPN heavily weights late-round games (champion = 320 pts). Optimize for ESPN, not log loss, when comparing strategies.

**3. TRAIN_YEARS in formula_model.py vs config.py are different.**
- `config.py` has `TRAIN_YEARS = [2013, 2014, 2015]` (an old split spec)
- `formula_model.py` has its own `TRAIN_YEARS = list(range(2013, 2022))` (9 seasons)
- The formula model uses ITS OWN constant. Don't accidentally import the config one.

**4. 2020 must always be excluded.**
No tournament was held in 2020 (COVID). Filter with `df["YEAR"] != 2020` or `y != 2020` in any year loop. Missing this adds a year with 0 games that silently skews results.

**5. C regularization is re-tuned each run via grid search.**
`run_temporal_cv()` grid searches C in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0] and returns the best C. The global `C_REGULARIZATION` constant is just the default — it gets overridden by the grid search result. Don't hard-trust the constant.

**6. `features_candidates.csv` must exist for the 7-feature model.**
If it's missing, `load_matchup_data()` falls back to `features_coaching.csv` which lacks `EFF_RATIO`. The model will silently drop to 6 features. Check the INFO log: it says which file was loaded.
