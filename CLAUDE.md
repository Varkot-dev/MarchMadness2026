# March Madness Bracket Prediction Model
## Claude Code Context File — Read This First

---

## What This Project Is

A machine learning system that predicts NCAA March Madness tournament outcomes
and generates optimized bracket submissions for a competitive bracket pool.

This is being built to beat a specific competitor (Michael Licamele) who is also
building a model using Claude Code. The edge comes from superior feature
engineering and a mathematically optimal multi-bracket submission strategy —
not just a better base model.

**Competition rules:**
- 3 brackets each, average score determines the winner
- ESPN bracket scoring (10pt R64, 20pt R32, 40pt S16, 80pt E8, 160pt F4, 320pt Champion)
- Deadline: March 20, 2026 (Selection Sunday)
- Pure model prediction — no manual overrides

---

## Engineering Standards

Act as a **senior software engineer** at all times. This means:

- Write modular, reusable functions — never monolithic scripts
- Every function gets a docstring: what it does, params, returns
- Prefer explicit and readable over clever one-liners
- Add error handling anywhere data fetching or file I/O can fail
- Before writing code, briefly explain your architectural approach
- If you see a better approach than what was asked for, say so
- Never hardcode file paths — use `pathlib.Path` and config constants
- Use type hints on all function signatures
- Log meaningful messages — not just "done" but what actually happened

---

## Project Structure

```
march-madness/
├── CLAUDE.md                  <- You are here
├── SKILLS.md                  <- Domain expertise, read this too
├── README.md
├── generate_2025_bracket.py   <- Run to regenerate predicted_bracket_2025.csv
├── data/
│   ├── raw/                   <- Never touch raw data after downloading
│   ├── processed/             <- Cleaned, feature-engineered data
│   └── external/kaggle/       <- Kaggle datasets (seeds, results, teams)
├── src/
│   ├── features/
│   │   ├── efficiency.py      <- TQS, Seed Divergence (clipped +-8), Luck correction
│   │   ├── momentum.py        <- Quality Momentum Score (Torvik game-by-game)
│   │   ├── coaching.py        <- Coaching Premium + CBB_TO_KAGGLE_NAMES map
│   │   └── starpower.py       <- NBA prospect depth scoring
│   ├── models/
│   │   ├── formula_model.py   <- PRIMARY: Explicit formula SCORE(T) = sum(wi * feature_i(T))
│   │   ├── shap_selector.py   <- Team-level SHAP feature selection (rounds_won target)
│   │   ├── simulator.py       <- Layer 2: Monte Carlo simulator (imports from config)
│   │   └── optimizer.py       <- Layer 3: Chalk/Medium/Chaos bracket optimizer
│   ├── utils/
│   │   └── team_names.py      <- build_daynum_to_round(), build_kaggle_to_cbb_map()
│   ├── evaluation/
│   │   └── backtest.py        <- Per-year backtest with dynamic round detection
│   └── output/
│       ├── brackets.py        <- Final bracket generation and export
│       └── visualizer.py      <- Matplotlib bracket visualizer
├── ui/
│   ├── app.py                 <- Flask web app (python -m ui.app -> :5050)
│   └── templates/index.html   <- Interactive bracket visualizer UI
├── tests/
│   └── test_features.py
├── config.py                  <- ALL shared constants live here — import, never redefine
└── requirements.txt
```

---

## config.py Is the Single Source of Truth

All shared constants are defined **once** in `config.py` and imported everywhere else.
Never redefine these in other files:

```python
ESPN_ROUND_POINTS     # {1:10, 2:20, 3:40, 4:80, 5:160, 6:320}
SEED_PAIRINGS         # [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
REGIONS               # ["South", "East", "West", "Midwest"]
FIRST_FOUR_2025       # Real Selection Sunday play-in matchups
SEED_DIVERGENCE_CLIP  # (-8, 8) — prevents extreme KenPom outliers
```

---

## Tech Stack

- **Python 3.11+**
- **pandas** — data wrangling
- **scikit-learn** — LR baseline, StandardScaler, calibration
- **XGBoost** — margin prediction model (install: `pip install xgboost`)
- **scipy** — sigmoid, normal CDF for margin-to-probability conversion
- **Flask** — web UI backend
- **matplotlib / seaborn** — visualization
- **pathlib** — all file paths (never use os.path)
- **pytest** — testing

---

## Critical Rules — Never Break These

1. **No temporal data leakage.** Always split train/test by season year.
   Never use random splits on historical game data. A game from 2020 must
   never appear in training data when evaluating 2019 performance.

2. **Optimize for bracket score, not accuracy.** A model that correctly
   identifies high-value upsets is more valuable than one that is 2% more
   accurate at picking chalk winners.

3. **Three brackets must have different entropy levels:**
   - Entry 1 (Chalk): maximize per-game probability via downstream expected score
   - Entry 2 (Medium): moderate upsets where P(dog) >= 35% AND Seed Divergence > 0
   - Entry 3 (Chaos): high-value upsets weighted by Seed Divergence (P(dog) >= 25% adjusted)

4. **Validate on held-out years before trusting any result.** Minimum 3 prior seasons.

5. **Import constants from config.py.** Never duplicate ESPN_ROUND_POINTS, SEED_PAIRINGS,
   or REGIONS in any other file.

---

## Data Sources

- **Kaggle March Mania** — primary training data, free, historical from 1985
- **KenPom** — efficiency metrics (kenpom.com), partial free access
- **Bart Torvik** — free KenPom alternative (barttorvik.com), scrapeable
- **Sports-Reference** — coaching records (sports-reference.com/cbb)
- **ESPN** — NBA draft board for prospect depth feature

---

## Current State (as of March 2026)

### Formula Model (`src/models/formula_model.py`) — PRIMARY PREDICTION ENGINE

The explicit learned formula that replaced the original chalk/medium/chaos heuristic approach:

```
SCORE(T) = sum(wi * feature_i(T))
P(A beats B) = sigmoid( SCORE(A) - SCORE(B) )
```

Weights are learned from logistic regression (elastic net, C=0.1) on 2013-2021 data.
Top features by impact (see `data/processed/formula_weights.csv`):
1. SEED_DIVERGENCE     — most important, positive weight (underseeded teams win more)
2. TRUE_QUALITY_SCORE  — second most important
3. ADJOE / ADJDE
4. WAB, SEED, COACH_PREMIUM, QMS, ORB, TOR

**Holdout results (2022-2024, never seen in training):**
- 2022: ESPN 720  — Houston predicted, Kansas won (wrong)
- 2023: ESPN 1320 — Connecticut predicted and won (correct)
- 2024: ESPN 1570 — Connecticut predicted and won (correct)
- Mean: 1203/1920 (63% of max), champion accuracy 2/3

### 2025 Live Prediction
- Trained on all available data: 2013-2024 (11 seasons, excl. 2020 COVID)
- Real Selection Sunday bracket hard-coded in `generate_2025_bracket.py`
- **Predicted 2025 champion: Florida**
- Notable upset picks: McNeese St. over Memphis, Baylor over Mississippi St.
- First Four results resolved via model win probability
- Output: `data/processed/predicted_bracket_2025.csv`

### Layer 1 — Win Probability (`src/models/formula_model.py`)
- **`win_probability.py` was deleted** — it was dead code bypassed by `run_full_pipeline()`
- Formula model is the sole prediction engine: `P(A beats B) = σ(score_A - score_B)`
- Shared utilities extracted to `src/utils/team_names.py`: `build_daynum_to_round()`, `build_kaggle_to_cbb_map()`
- Features: `TRUE_QUALITY_SCORE` (corr=0.596) + `SEED_DIVERGENCE` (corr=0.313) with rounds_won
- WAB investigated as 3rd feature but r=0.93 with TQS causes sign-flip collinearity — not added
- Tournament experience (prior game minutes) is the likely fix for 2022 Houston/Kansas miss

### Layer 2 — Monte Carlo Simulator (`src/models/simulator.py`)
- Fully dynamic: loads bracket from Kaggle MNCAATourneySeeds.csv for any year
- ESPN_ROUND_POINTS and SEED_PAIRINGS imported from config (not redefined locally)
- 10,000 simulations, selects p90-maximizing bracket

### Layer 3 — Optimizer (`src/models/optimizer.py`)
- Was completely broken (missing imports, wrong function signatures) — now fixed
- Imports ESPN_ROUND_POINTS, REGIONS, FIRST_FOUR_2025 from config
- `_build_bracket_structure_2025()` provides hardcoded 2025 Selection Sunday fallback
  when Kaggle seed file doesn't have 2025 data yet
- Run: `python -m src.models.optimizer`

### Feature Engineering Notes
- **Seed Divergence** clipped to +-8 to prevent extreme KenPom outliers from
  destabilizing LR weights (raw values could previously reach +-49)
- **True Quality Score** = AdjEM - (Luck * 0.4), falls back to ADJOE-ADJDE if no KenPom
- **QMS** from Torvik game-by-game data; years without Torvik CSV get QMS=0 (known limitation)
- **Coach Premium** uses CBB_TO_KAGGLE_NAMES map in coaching.py for team name bridging

### Web UI (`ui/app.py` + `ui/templates/index.html`)
- Launch: `python -m ui.app` -> http://localhost:5050
- Regions color-coded (South=blue, East=purple, West=orange, Midwest=teal)
- Green/red left-border stripes on team slots for correct/wrong picks (holdout years)
- Click any game -> sidebar: animated probability bar, model vs actual, feature comparison table
- Formula button -> model weights as bar charts + per-year CV accuracy
- `_NaNSafeEncoder` on the Flask app handles float NaN->null globally in all JSON responses
- Invalid year requests return HTTP 400 with a clear list of available years

### Key Processed Files
- `data/processed/features_coaching.csv`              — full feature matrix (2013-2025)
- `data/processed/formula_weights.csv`                — learned formula coefficients
- `data/processed/formula_cv_results.csv`             — temporal CV per fold
- `data/processed/win_prob_matrix_2025.csv`           — 68x68 pairwise probabilities
- `data/processed/predicted_bracket_{2022,2023,2024,2025}.csv` — bracket predictions

---

## Known Gaps / Next Improvements

- **XGBoost not installed on this machine** — LR-only currently; `pip install xgboost` to enable
- **QMS degrades silently** when Torvik CSV is missing for a year (gets QMS=0 with a log warning)
- **No tournament experience feature** — prior March Madness minutes played is a known
  predictor not yet implemented
- **No unit tests for models** — tests/ only covers features; simulator/optimizer/formula_model
  have no test coverage
- **2022 champion badly wrong — root cause identified:** Houston had TQS=27.70/SEED_DIV=+4
  vs Kansas TQS=27.47/SEED_DIV=0. WAB was investigated (Kansas 10.4 vs Houston 6.2) but
  r=0.93 with TQS causes sign-flip collinearity — adding WAB doesn't flip the pick.
  Tournament experience is the likely fix: Kansas had 61 prior tourney games vs Houston's 11.
  This will be addressed by `feature/tourney-experience` branch (other agent).

---

## Claude Code Skills (`.claude/skills/`)

Invoke with `/skill-name` or by describing what you want:

| Skill | Trigger phrases | What it does |
|---|---|---|
| `add-feature` | "add feature", "new feature" | Scaffolds a new feature module under `src/features/`, wires into `candidates.py` |
| `run-backtest` | "backtest", "validate", "CV" | Validates model correctly, interprets ESPN vs CV metrics |
| `model-experiment` | "experiment", "try model" | Scaffolds experiment scripts with baseline comparison |
| `add-feature-test` | "write tests", "add tests" | Scaffolds unit + integration tests for feature modules |
| `check-temporal-leakage` | "check leakage", "data leakage" | Audits code for temporal data leakage |
| `fix-team-mapping` | "fix team mapping", "unmapped teams", "team name mismatch" | Diagnoses unmapped CBB↔Kaggle team names and patches `CBB_TO_KAGGLE_NAMES` |

---

## How to Work With Me

- Read SKILLS.md first for domain context before building any feature
- Always tell me what you are about to do before doing it
- If something will take multiple steps, outline the steps first
- When you write a function, also write a quick test or usage example
- If I ask for something that conflicts with the critical rules above, flag it
- Import constants from config.py — never redefine them
