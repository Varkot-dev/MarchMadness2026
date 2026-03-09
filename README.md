# March Madness Predictor 2026

A machine learning system that predicts NCAA March Madness tournament outcomes and generates three optimized bracket submissions using a three-layer architecture: win probability model → Monte Carlo simulator → multi-bracket optimizer.

---

## Architecture

The system is built in three independent layers, each with a clean input/output contract.

```
Layer 1: Win Probability Engine
  Input:  Two teams + their feature vectors
  Output: P(team A beats team B) for every tournament matchup pair
  Files:  src/models/win_probability.py

Layer 2: Monte Carlo Tournament Simulator
  Input:  68-team win probability matrix
  Output: P(team reaches each round) across 10,000 simulated tournaments
  Files:  src/models/simulator.py

Layer 3: Multi-Bracket Optimizer
  Input:  P(reach) table + seed divergence scores
  Output: Three brackets at different entropy levels
  Files:  src/models/optimizer.py
```

---

## Features

Ten engineered features are computed per team per season. Each is differenced (A − B) to form the matchup feature vector.

| Feature | Description |
|---|---|
| **True Quality Score** | `AdjEM − (Luck × 0.4)` — strips luck from efficiency to reveal actual team strength |
| **Seed Divergence** | KenPom implied seed minus actual seed — identifies underseeded sleepers |
| **Quality Momentum Score** | Last-10-game win quality, weighted by opponent KenPom rank (top-25=10pts, top-50=7pts, top-100=4pts) |
| **Coaching Premium** | Coach career tournament wins above seed expectation |
| **AdjOE** | Adjusted offensive efficiency (points per 100 possessions) |
| **AdjDE** | Adjusted defensive efficiency (points allowed per 100 possessions) |
| **WAB** | Wins above bubble — measures schedule-adjusted performance |
| **TOR** | Turnover rate |
| **ORB** | Offensive rebound rate |
| **SEED** | Raw seed number |

---

## Three-Bracket Strategy

Submitting three identical brackets is mathematically suboptimal under ESPN's top-heavy scoring system (1 / 2 / 4 / 8 / 16 / 32 points per round). The optimal strategy diversifies entropy across entries.

| Bracket | Strategy | How picks are made |
|---|---|---|
| **Chalk** | Maximize per-game expected score | Pick the team with higher downstream expected ESPN score at every node |
| **Medium** | Moderate upsets | Take the underdog when P(dog wins) ≥ 35% AND the committee underseeded them (Seed_Divergence > 0) |
| **Chaos** | High-variance lottery | Take the underdog when P(dog wins) ≥ 25% − 4%×seed_divergence, explicitly targeting mispriced matchups |

**2025 results (10,000 simulations, seed=42):**

| Bracket | E[ESPN Score] | Champion |
|---|---|---|
| Chalk | 1782 | Duke |
| Medium | 1780 | Duke |
| Chaos | 1651 | Duke |

The Chaos bracket deliberately trades ~130 expected points for a higher ceiling — it is the lottery ticket entry.

---

## Validation

All metrics are computed from held-out predictions only. No number in this table is hardcoded.

| Method | Accuracy | Log Loss | Brier Score |
|---|---|---|---|
| LOSO-CV (Leave-One-Season-Out) | 74.0% | 0.510 | 0.170 |
| Temporal CV (strict, ≥3 prior seasons) | 73.7% | 0.514 | — |

**Temporal CV per year (LR model):**

| Year | Train Seasons | Accuracy | Log Loss |
|---|---|---|---|
| 2016 | 3 | 73.3% | 0.475 |
| 2017 | 4 | 71.4% | 0.509 |
| 2018 | 5 | 73.2% | 0.552 |
| 2019 | 6 | 79.3% | 0.417 |
| 2021 | 7 | 76.8% | 0.505 |
| 2022 | 8 | 73.7% | 0.528 |
| 2023 | 9 | 72.6% | 0.562 |
| 2024 | 10 | 69.1% | 0.567 |

---

## Project Structure

```
march-madness/
├── config.py                        ← All constants, paths, hyperparameters
├── requirements.txt
├── data/
│   ├── raw/                         ← Source CSVs (cbb.csv, cbb25.csv)
│   ├── processed/                   ← Engineered features, model outputs
│   │   ├── features_coaching.csv    ← Final feature matrix (2013–2024)
│   │   ├── win_prob_matrix_2025.csv ← 68×68 pairwise win probabilities
│   │   ├── three_brackets_2025.csv  ← Chalk / Medium / Chaos bracket picks
│   │   ├── calibration_curve.png    ← Model calibration (LOSO-CV)
│   │   └── score_distribution.png  ← Bracket score distribution (10k sims)
│   └── external/                    ← KenPom, Torvik, Kaggle, coaching data
├── src/
│   ├── data/
│   │   ├── fetch_kenpom.py          ← KenPom efficiency data loader
│   │   ├── fetch_torvik.py          ← Bart Torvik game-by-game loader
│   │   ├── fetch_coaching.py        ← Coaching records loader
│   │   └── preprocess.py            ← Cleaning and validation
│   ├── features/
│   │   ├── efficiency.py            ← True Quality Score, Seed Divergence
│   │   ├── momentum.py              ← Quality Momentum Score
│   │   └── coaching.py              ← Coaching Premium multiplier
│   ├── models/
│   │   ├── win_probability.py       ← Layer 1: logistic regression + calibration
│   │   ├── simulator.py             ← Layer 2: Monte Carlo bracket simulation
│   │   └── optimizer.py             ← Layer 3: multi-bracket optimizer
│   └── evaluation/                  ← Backtesting utilities (in progress)
├── notebooks/
│   └── exploration/                 ← EDA only
└── tests/
    ├── test_efficiency.py
    ├── test_momentum.py
    └── test_coaching.py
```

---

## Data Sources

| Source | Used For | Access |
|---|---|---|
| [Kaggle March Mania 2025](https://www.kaggle.com/competitions/march-machine-learning-mania-2025) | Historical tournament results (1985–2024) | Free |
| [KenPom](https://kenpom.com) | AdjEM, AdjOE, AdjDE, Luck, WAB | Partial free access |
| [Bart Torvik](https://barttorvik.com) | Game-by-game data for momentum features | Free, scrapeable |
| [Sports-Reference CBB](https://www.sports-reference.com/cbb) | Coaching records | Free |

---

## Setup

```bash
git clone https://github.com/Varkot-dev/MarchMadness2026.git
cd MarchMadness2026
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the Kaggle dataset and place CSVs in `data/external/kaggle/`. Then run each layer in order:

```bash
# Layer 1: train model and build win probability matrix
python -m src.models.win_probability

# Layer 2: run Monte Carlo simulations
python -m src.models.simulator

# Layer 3: generate three optimized brackets
python -m src.models.optimizer
```

Run tests:

```bash
pytest tests/
```

---

## Key Design Decisions

**Why temporal CV instead of random CV?**
Tournament games from 2023 must never appear in training data when evaluating 2019 performance. Random splits cause temporal data leakage that inflates accuracy estimates. Every evaluation split in this project is by season year.

**Why three brackets instead of one?**
With ESPN's exponential scoring (Round 1 = 1pt, Champion = 32pt) and three allowed entries averaged together, the optimal strategy is not to maximize expected score per bracket but to maximize the expected score of the *best* bracket across all three. This requires entropy diversification — one safe pick, one moderate upset pick, one high-variance pick.

**Why optimize for bracket score instead of accuracy?**
A model that correctly identifies a 12-over-5 upset (1 pt) and correctly picks the champion (32 pts) is worth far more than a model that is 2% more accurate at calling chalk first-round games. These are different objectives. This project optimizes for expected bracket score directly.

**Why Seed Divergence matters for upset picking?**
The selection committee weights win-loss record heavily. Teams that played brutal schedules may have more losses but be much stronger than their seed implies. Seed Divergence = KenPom implied seed − actual seed. A positive value means the model thinks the team is better than where they are seeded — the strongest signal for identifying mispriced matchups.
