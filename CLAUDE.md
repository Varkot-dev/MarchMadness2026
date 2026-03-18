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

### Primary Model (`src/models/formula_model_new.py`) — NEW 16-YEAR PIPELINE

```
SCORE(T) = w1*WAB(T) + w2*TALENT(T) + w3*KADJ_O(T)
P(A beats B) = sigmoid( SCORE(A) - SCORE(B) )
```

Trained on 2008–2024 (16 seasons, excl. 2020). Features selected by SHAP + holdout grid search.

Top features: WAB (77.3% weight) > KADJ O (17.5%) > TALENT (5.2%)

**Holdout results (2022-2024, new pipeline):**
- 2022: ESPN 1180 — **Kansas predicted and won (correct ✓)**
- 2023: ESPN 450  — Alabama predicted, Connecticut won (wrong — no Luck correction in Barttorvik)
- 2024: ESPN 1280 — **Connecticut predicted and won (correct ✓)**
- Mean: 970/1920, champion accuracy 2/3

**Full 8-year backtest (2016-2024 excl. 2020):** mean 902/1920, 3/8 champions correct

### 2026 Live Prediction
- Trained on all 2008–2024 data (16 seasons, excl. 2020)
- **Predicted 2026 champion: Duke** (1-seed South, #1 KADJ EM 38.9, TALENT 91.2)
- Final Four: Duke vs Houston, Arizona vs Michigan
- Championship: Duke over Arizona (53.6%)
- Output: `data/processed/predicted_bracket_2026_new.csv`

### Layer 1 — Win Probability (`src/models/formula_model_new.py`)
- Primary: `formula_model_new.py` — WAB+TALENT+KADJ O on new 16-year dataset
- Legacy: `formula_model.py` — TQS+SEED_DIVERGENCE on features_coaching.csv (kept for reference)
- `win_probability.py` was deleted — dead code bypassed by `run_full_pipeline()`
- Shared utilities: `src/utils/team_names.py` — `build_daynum_to_round()`, `build_kaggle_to_cbb_map()`

### Data Pipeline (New)
- `src/features/new_data_loader.py` — loads 2008–2026 KenPom+Barttorvik+EvanMiya+Resumes
- `src/features/new_matchup_builder.py` — 1047 games via Kaggle explicit WTeamID/LTeamID pairs
- `src/models/shap_new_data.py` — SHAP feature selection on 16-year dataset
- Team names consistent across all new CSVs — zero mapping issues within new dataset
- Kaggle bridge: 254/263 unique teams mapped (96.6% coverage)

### Layer 2 — Monte Carlo Simulator (`src/models/simulator.py`)
- Fully dynamic: loads bracket from Kaggle MNCAATourneySeeds.csv for any year
- ESPN_ROUND_POINTS and SEED_PAIRINGS imported from config (not redefined locally)
- 10,000 simulations, selects p90-maximizing bracket

### Layer 3 — Optimizer (`src/models/optimizer.py`)
- Imports ESPN_ROUND_POINTS, REGIONS, FIRST_FOUR_2025 from config
- Run: `python -m src.models.optimizer`

### Web UI (`ui/app.py` + `ui/templates/index.html`)
- Launch: `python -m ui.app` -> http://localhost:5050
- Available years: 2022, 2023, 2024, 2025, 2026
- 2022-2024: holdout predictions (new pipeline), green/red correct/wrong indicators
- 2026: live prediction — Duke champion
- Click any game -> sidebar: win probability, feature comparison (WAB, TALENT, KADJ O)
- Formula button -> model weights bar chart + per-year CV accuracy (12 folds)

### Key Processed Files
- `data/processed/features_new.csv`                   — 1215 rows, 2008-2026, 52 columns
- `data/processed/matchups_new.csv`                   — 2094 rows (1047 games, 2008-2024)
- `data/processed/formula_new_weights.csv`            — WAB/TALENT/KADJ O weights
- `data/processed/formula_new_cv_results.csv`         — 12-fold temporal CV results
- `data/processed/predicted_bracket_{2022,2023,2024}_new.csv` — holdout predictions
- `data/processed/predicted_bracket_2026_new.csv`     — live 2026 prediction
- `data/processed/features_coaching.csv`              — legacy feature matrix (2013-2025)

---

## Known Gaps / Next Improvements

- **2023 UConn miss** — Alabama dominates on WAB/TALENT/KADJ O; UConn's edge is defense
  (KADJ D). No Luck correction in Barttorvik data means UConn's "overachievement" isn't
  captured. KenPom Luck metric would fix this but isn't in the new dataset.
- **No unit tests for models** — tests/ only covers features
- **Simulator/optimizer not yet wired to new model** — still uses old formula_model.py
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
