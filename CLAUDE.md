# March Madness Bracket Prediction Model
## Claude Code Context File — Read This First

---

## What This Project Is

A machine learning system that predicts NCAA March Madness tournament outcomes
and generates three optimized bracket submissions for a competitive bracket pool.

This is being built to beat a specific competitor (Michael Licamele) who is also
building a model using Claude Code. The edge comes from superior feature
engineering and a mathematically optimal multi-bracket submission strategy —
not just a better base model.

**Competition rules:**
- 3 brackets each, average score determines the winner
- ESPN bracket scoring (1pt R1, 2pt R2, 4pt S16, 8pt E8, 16pt F4, 32pt Champion)
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
├── CLAUDE.md                  ← You are here
├── SKILLS.md                  ← Domain expertise, read this too
├── README.md
├── data/
│   ├── raw/                   ← Never touch raw data after downloading
│   ├── processed/             ← Cleaned, feature-engineered data
│   └── external/              ← KenPom, Kaggle datasets
├── src/
│   ├── data/
│   │   ├── fetch.py           ← Data downloading and caching
│   │   └── preprocess.py      ← Cleaning and validation
│   ├── features/
│   │   ├── efficiency.py      ← AdjEM, True Quality Score, Luck correction
│   │   ├── momentum.py        ← Quality Momentum Score
│   │   ├── coaching.py        ← Coaching Premium multiplier
│   │   └── starpower.py       ← NBA prospect depth scoring
│   ├── models/
│   │   ├── win_probability.py ← Layer 1: P(team A beats team B)
│   │   ├── simulator.py       ← Layer 2: Monte Carlo tournament simulation
│   │   └── optimizer.py       ← Layer 3: Multi-bracket DP optimizer
│   ├── evaluation/
│   │   └── backtest.py        ← Temporal cross-validation by season year
│   └── output/
│       └── brackets.py        ← Final bracket generation and export
├── notebooks/
│   └── exploration/           ← EDA only, no production code here
├── tests/
│   └── test_features.py
├── config.py                  ← All constants, paths, hyperparameters
└── requirements.txt
```

---

## Tech Stack

- **Python 3.11+**
- **pandas** — data wrangling
- **scikit-learn** — baseline models, preprocessing
- **XGBoost / LightGBM** — primary model
- **scipy** — statistical utilities
- **matplotlib / seaborn** — visualization
- **requests / beautifulsoup4** — data fetching if needed
- **pathlib** — all file paths (never use os.path)
- **pytest** — testing

---

## Critical Rules — Never Break These

1. **No temporal data leakage.** Always split train/test by season year.
   Never use random splits on historical game data. A game from 2020 must
   never appear in training data when evaluating 2019 performance.

2. **Optimize for bracket score, not accuracy.** A model that correctly
   identifies high-value upsets is more valuable than one that is 2% more
   accurate at picking chalk winners. Keep this in mind when evaluating models.

3. **Three brackets must have different entropy levels.** Never generate
   three nearly-identical brackets. The three entries must be:
   - Entry 1 (Chalk): maximize per-game probability
   - Entry 2 (Medium): moderate upsets where model diverges from seed
   - Entry 3 (Chaos): high-value upsets where model and market diverge most

4. **Validate on held-out years before trusting any result.** If a model
   cannot be validated on at least 3 prior tournaments, do not trust it.

---

## Data Sources

- **Kaggle March Mania** — primary training data, free, historical from 1985
  URL: https://www.kaggle.com/competitions/march-machine-learning-mania-2025
- **KenPom** — efficiency metrics (kenpom.com), partial free access
- **Bart Torvik** — free KenPom alternative (barttorvik.com), scrapeable
- **Sports-Reference** — coaching records, historical data (sports-reference.com/cbb)
- **ESPN** — NBA draft board for prospect depth feature

---

## Where We Are

The research phase is complete. Five analytical agents identified the
key differentiating features and the mathematical framework for
multi-bracket optimization. The SKILLS.md file contains full domain
expertise. The next step is building the data pipeline and feature
engineering layer.

---

## How to Work With Me

- When I ask you to build something, read SKILLS.md first for domain context
- Always tell me what you're about to do before doing it
- If something will take multiple steps, outline the steps first
- When you write a function, also write a quick test or usage example
- If I ask for something that conflicts with the critical rules above, flag it
