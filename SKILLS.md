# SKILLS.md — March Madness Domain Expertise
## Read this before building any feature or model component

---

## The Core Problem (Read This Carefully)

Most March Madness models solve the wrong problem. They optimize for
**prediction accuracy** (what % of games did you pick correctly).

This model solves the right problem: **maximize expected score of the
best bracket across three entries under ESPN's top-heavy scoring system.**

These are fundamentally different objectives. A model that correctly
identifies a 12-over-5 upset worth 10 points AND correctly picks the
champion worth 320 points is dramatically more valuable than a model
that is 3% more accurate at picking chalk Round 1 games.

**ESPN Bracket Scoring (correct values):**
| Round | Points |
|-------|--------|
| R64 (Round 1) | 10 |
| R32 (Round 2) | 20 |
| S16 (Sweet 16) | 40 |
| E8 (Elite 8) | 80 |
| F4 (Final Four) | 160 |
| Champion | 320 |
Maximum possible score: **1920 points** (if all 63 games correct).

Always keep this in mind when making modeling decisions.

---

## The Three-Layer Architecture

### Layer 1: Win Probability Engine
**What it does:** Given any two teams, output P(team A wins) on a neutral court.

**Inputs:** Feature vectors for both teams (see Feature Engineering below)
**Models:** Logistic Regression (primary) + XGBoost margin model (secondary)
**Formula model alternative:** `P(A beats B) = σ(SCORE_A − SCORE_B)` where score is weighted feature sum
**Training:** Kaggle March Mania historical data, 2013–2024 (excl. 2020 COVID year)
**Validation:** ALWAYS split by season year, never random split
**Output:** Float [0.0, 1.0] representing win probability

**Implemented files:**
- `src/models/formula_model.py` — PRIMARY: formula weights, simulate_bracket(), run_full_pipeline()
- `src/utils/team_names.py` — shared utilities: build_daynum_to_round(), build_kaggle_to_cbb_map()
- `src/models/win_probability.py` — DELETED (was dead code bypassed by formula_model)

**Current performance (temporal CV, 2016–2024):**
- New model (WAB + TALENT + KADJ O, features_new.csv 2008–2024): **70.7% accuracy, 0.563 log loss** (8 CV folds)
- Holdout ESPN: 1120 (2022 Kansas ✓), 480 (2023 UConn miss), 1380 (2024 UConn ✓) — mean **993/1920** on 2022-24
- Old model (TQS + SEED_DIVERGENCE, features_coaching.csv 2013–2021): 77.5% acc, 0.493 ll
- Old holdout ESPN: 610 (2022), 1200 (2023), 1300 (2024) — mean 1037/1920
- 2023 miss explained: no Luck correction in new data → Alabama ranked above UConn on raw metrics

### Layer 2: Monte Carlo Tournament Simulator
**What it does:** Simulate the full 64-team bracket thousands of times
to build a probability distribution over all possible outcomes.

**Process:**
1. Load the actual bracket seedings once announced
2. For every possible matchup, compute win probability from Layer 1
3. Run 10,000+ simulations — each game sampled from win probability
4. Track: P(team reaches each round), P(team wins championship)
5. Output: full probability map, not a single bracket

**Key insight:** This captures path dependency. A team might face an
easy path to the Elite 8 based on how the bracket sets up. The simulator
accounts for this automatically.

**Implemented file:** `src/models/simulator.py`

### Layer 3: Multi-Bracket DP Optimizer
**What it does:** Given the probability map from Layer 2, construct three
brackets that maximize expected best-bracket ESPN score.

**The math:** With top-heavy scoring and multiple entries, submitting
three identical brackets is suboptimal. The optimal strategy diversifies
entropy across entries.

**Three bracket types:**
- **Chalk bracket:** Pick highest-probability winner at each game.
  Safe floor, mid-range ceiling. Always include at least one 1-seed
  in the Final Four.
- **Medium bracket:** Pick moderate upsets where model probability
  significantly exceeds seed expectation. Target 6-11 and 5-12 games
  where True Quality Score diverges from seed.
- **Chaos bracket:** Pick high-value upsets where model probability
  AND market (betting odds) both undervalue the underdog. Low floor,
  high ceiling. This is the lottery ticket entry.

**Implemented file:** `src/models/optimizer.py`

---

## Validated Holdout Results

| Year | Predicted Champion | Actual | ESPN Score | % of Max |
|------|--------------------|--------|------------|----------|
| 2022 | Houston | Kansas ✗ | 720/1920 | 37.5% |
| 2023 | **Connecticut** | Connecticut ✓ | 1320/1920 | 68.8% |
| 2024 | **Connecticut** | Connecticut ✓ | 1570/1920 | 81.8% |
| Mean | — | 2/3 correct | 1203/1920 | 62.7% |

CV performance (temporal, training years 2016–2021): **77.5% accuracy, 0.468 log loss**

**2025 Live Prediction:** Model predicts **Florida** as champion (trained on all 2013–2024 data).

---

## Feature Engineering — Full Specification

### Feature 1: True Quality Score (PRIMARY)
**Formula:** `True_Quality = AdjEM - (Luck * 0.4)`
**Source:** KenPom
**Why:** AdjEM is the strongest raw predictor, but teams with high Luck
ratings won more games than their efficiency deserved. In the tournament,
luck regresses toward zero. Stripping it out reveals true team strength.
**Implementation note:** The 0.4 coefficient is a starting estimate.
Calibrate on held-out historical seasons.

### Feature 2: Seed Divergence Score
**Formula:** `Seed_Divergence = actual_seed - KenPom_implied_seed`
**Source:** KenPom rank converted to implied seed
**Why:** The selection committee uses win-loss record heavily. Teams that
played brutal schedules may have more losses but be much stronger than
their seed. Positive divergence = underseeded = potential sleeper.
**Implementation:** Convert KenPom rank to approximate seed by binning
(ranks 1-4 → implied seed 1, ranks 5-8 → implied seed 2, etc.)
**Sign convention (important):** `actual_seed - implied_seed`. A team ranked
#4 by KenPom but given seed 4 by the committee has implied_seed=1, so
divergence = 4 - 1 = +3 (positive = underseeded = upset threat). The old
formula was inverted and has been corrected in the codebase.
**Validated weight:** +0.160 in the final model (positive = underseeded teams win more)
**Clipping:** Raw values can reach ±49; clip to ±8 (2 seed lines) to prevent LR instability.
`SEED_DIVERGENCE_CLIP = (-8, 8)` lives in `config.py`.

### Feature 3: Quality Momentum Score
**Formula:** For last 10 games before tournament:
```
QMS = sum(win * weight) where:
  win vs top-25 KenPom team  → weight = 10
  win vs top-50 KenPom team  → weight = 7
  win vs top-100 KenPom team → weight = 4
  win vs below-100 team      → weight = 1
  loss                       → weight = 0
```
**Source:** Bart Torvik game-by-game data
**Why:** Standard momentum (win streak) is misleading. Winning 8 of 10
against weak opponents tells you nothing about tournament readiness.
Winning 6 of 10 against tournament-caliber teams tells you everything.
**Flag:** Teams with QMS > 60 entering tournament historically outperform seed.

### Feature 4: Coaching Premium
**Formula:** `Coach_Premium = career_tournament_wins - expected_wins_by_seed`
**Source:** Sports-Reference coaching records
**Why:** Tom Izzo is 7 standard deviations above average at outperforming
his seed. This is not noise — it is a real, computable signal.
Apply as a probability multiplier on specific matchups where coaching
gap is large (e.g., Izzo as underdog).
**Implementation:** Build a lookup table of active coaches with their
historical tournament performance vs seed expectation. Apply as a
scalar multiplier to base win probability.

### Feature 5: NBA Prospect Depth Score
**Formula:** `Prospect_Depth = sum(weight_i) for all players on roster`
where weight = 10 if top-15 prospect, 7 if top-30, 4 if top-50, 0 otherwise
**Source:** ESPN NBA Draft Board
**Why:** Teams with ONE elite prospect are fragile in single elimination.
If that player has an off game, the team has no fallback. Teams with
2-3 top-50 prospects are resilient and often underseeded because the
committee focuses on record, not talent depth.
**Flag:** Single-star teams (one top-15 + no other top-50 prospects)
as high-variance picks in the chaos bracket only.

### Feature 6: Prior March Madness Experience
**Formula:** `MM_Experience = sum of NCAA tournament minutes played by
all current roster members in prior seasons`
**Source:** Sports-Reference player game logs
**Why:** Research across 693 tournament games (2007-2017) shows that
raw class rank (senior-heavy teams) has NO significant advantage.
But actual prior March Madness minutes played IS significant,
specifically in rounds 3 and beyond (Sweet 16+).
**Implementation:** Pull each player's career tournament minutes from
Sports-Reference. Sum across the current roster.

### Feature 7: Adjusted Defensive Efficiency (AdjDE / KENPOM_DRTG)
Partially independent from TQS. Use selectively — test for collinearity before including.

### Features NOT yet built (future work)
- **2-point % allowed** — interior defense, most predictive in recent models
- **Free throw rate mismatch** — if team relies on FTs vs opponent that doesn't foul
- **Non-conference SOS (NCSOS)** — reveals how seriously coach prepares for March

### Standard KenPom Features (Use Selectively — Collinearity Warning)
- AdjT (Adjusted Tempo)
- 2-point shooting % allowed (interior defense — most predictive in recent models)
- Free throw rate mismatch (if team relies on FTs vs opponent that doesn't foul)
- Non-conference SOS (NCSOS) — reveals how seriously coach prepares for March

**Do NOT add AdjO, WAB, or SEED alongside TRUE_QUALITY_SCORE.** These are
collinear (r > 0.83 with TQS) and adding them hurts model accuracy with small
datasets. Temporal CV showed dropping them improved accuracy from 73.1% → 77.5%.
With more training data (e.g., 2026+) this may change — re-test annually.

---

## Seeding Patterns to Encode

These are statistically validated patterns. Do not ignore them:

| Matchup | Historical upset rate | Model adjustment |
|---------|----------------------|------------------|
| 12 vs 5 | 35% (12-seed wins) | Flag when 12-seed has positive Seed_Divergence |
| 11 vs 6 | ~37% currently | 11-seeds have winning record in recent years |
| 10 vs 7 | ~39% | Near coin flip — let efficiency metrics decide |
| 9 vs 8  | ~50% | Treat as coin flip, efficiency only |
| 15 vs 2 | ~10% | Include small probability in chaos bracket |
| 16 vs 1 | ~2%  | Almost never, but UMBC 2018 happened |

**Key insight on 5-12:** The 35% number is aggregate. The REAL signal
is when the 12-seed profiles 2+ seed lines better in KenPom than their
actual seed. These are mispriced matchups, not random upsets.

---

## What the Champion Profile Looks Like

Based on historical data since 2002:
- Top-20 in BOTH AdjO and AdjD (almost always — only 2 exceptions)
- KenPom rank top-4 (effectively a 1-seed in efficiency even if not in seeding)
- True Quality Score > 25
- At least one elite guard (11 of last 15 MOPs have been guards)
- Not heavily reliant on 3-point shooting for offense (high variance in tournament)
- Coach with positive tournament premium

**2025 model pick:** Florida (1-seed West) — top KenPom rank, strong TQS, positive coaching premium.

Use this profile to weight championship probability in the simulator.

---

## Backtesting Requirements

Before trusting any model output:

1. **Temporal split:** Train on years N-1 and before, test on year N
2. **Minimum validation:** Test on at least 3 separate tournament years
3. **Metric to optimize:** Expected bracket score, not log-loss or accuracy
4. **Baseline to beat:** Simple seed-based model (pick higher seed every game)
   typically scores ~120/192 possible points under old scoring. Any model must beat this.
5. **Secondary metric:** Correct Final Four picks per tournament year
6. **DayNum → round mapping:** Never hardcode. Use `_build_daynum_to_round()` in
   `src/evaluation/backtest.py` — it infers rounds dynamically from game counts
   per day and works correctly across all years including the 2021 bubble.

---

## Common Mistakes — Flag These If You See Them

- Using random train/test split on game data (temporal leakage)
- Treating AdjEM as ground truth without luck correction
- Generating three nearly-identical brackets
- Optimizing for accuracy instead of expected bracket score
- Ignoring path dependency (who a team faces in round 3 matters for round 1)
- Using class rank as an experience proxy (use actual tournament minutes)
- Picking upsets randomly vs picking upsets where Seed_Divergence is positive
- **Adding collinear features (AdjO, WAB, SEED) alongside TQS** — proven to hurt accuracy
- **Hardcoding DayNum→round mappings** — they shift across years, always infer dynamically
- **Using `implied - actual` for Seed Divergence** — sign is inverted; use `actual - implied`
- **Reporting ESPN scores without validating the round detection** — off-by-one DayNum bugs
  make scores look much higher than they are (we caught a 110-point inflation in 2024)
- **Not clipping Seed Divergence** — raw values hit ±49, must clip to ±8 (`SEED_DIVERGENCE_CLIP` in config.py)
- **Hardcoding sigma=10.5 for XGBoost margin model** — fit sigma from training data per fold using `fit_margin_sigma()`
- **Defining ESPN_ROUND_POINTS, SEED_PAIRINGS, REGIONS locally** — import from `config.py` (single source of truth)
- **Using loop over DataFrame rows for win probability matrix** — vectorize: build all pairwise diffs, single `predict_proba` call
- **Expecting Kaggle seed data for current year** — Kaggle lags by months; hardcode bracket from Selection Sunday

---

## Data Pipeline Notes

**Kaggle dataset columns you need:**
- TeamID, Season, Seed, WTeamID, LTeamID, WScore, LScore

**KenPom columns you need:**
- AdjEM, AdjO, AdjD, AdjT, Luck, NCSOS

**Bart Torvik advantage:**
- Has game-by-game data accessible for free
- Use for Quality Momentum Score calculation (last 10 games per team)
- URL pattern: barttorvik.com (check current URL structure)

**Sports-Reference:**
- Coaching records: sports-reference.com/cbb/coaches
- Player game logs: sports-reference.com/cbb/players
- Be respectful with scraping — add delays between requests

**Team name bridging:**
- Kaggle uses its own TeamID system; CBB dataset uses different names
- 166 games dropped historically due to name mismatch — maintain a name bridge table
- Always use Kaggle TeamID as the primary key when joining across datasets

---

## File Naming Conventions

- Features: `feature_{name}.py` in src/features/
- All DataFrames: snake_case column names
- Team IDs: always use Kaggle TeamID as the primary key for joining
- Season year: always the year the tournament was played (2024 = 2023-24 season)

---

## Config.py — Single Source of Truth

All shared constants live in `config.py`. Never redefine locally:

```python
ESPN_ROUND_POINTS = {1: 10, 2: 20, 3: 40, 4: 80, 5: 160, 6: 320}
SEED_PAIRINGS = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
REGIONS = ["South", "East", "West", "Midwest"]
SEED_DIVERGENCE_CLIP = (-8, 8)
FIRST_FOUR_2025 = [...]  # 6 play-in games with real Selection Sunday matchups
```
