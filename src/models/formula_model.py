"""
formula_model.py — Learn the explicit mathematical formula from all historical
data, then generate predictions for held-out years using that formula.

Approach:
  1. Gather all tournament games from 2013 → 2021 as training data.
     (Leave 2022, 2023, 2024 completely untouched as prediction targets.)
  2. Run temporal CV on 2013–2021 to validate that the weights are stable —
     i.e. that the formula isn't overfit to any single season.
  3. Fit the FINAL model on all 2013–2021 data (9 seasons, ~560 games).
  4. Extract the explicit formula: score(team) = w1*TQS + w2*QMS + ... + b
     These weights ARE the model. Print them. Save them.
  5. For each holdout year (2022, 2023, 2024):
     - Compute each team's model score using the formula.
     - For each matchup, P(A beats B) = sigmoid(score_A - score_B).
     - Simulate the bracket deterministically (always pick the higher-prob team).
     - Score against actual results.
     - Save a full matchup-by-matchup result CSV.

The formula is a logistic regression over feature differences, which means it
reduces to a linear score per team:

    team_score(T) = w1*TQS(T) + w2*QMS(T) + w3*COACH(T) - w4*SEED(T) + ...

    P(A beats B) = σ( score(A) - score(B) )   where σ is the sigmoid function

This is the actual learned equation. Every weight is derived from backtesting
over 9 seasons of tournament data with strict temporal ordering.

Output files (data/processed/):
  - formula_weights.csv         — the explicit formula weights
  - formula_cv_results.csv      — temporal CV metrics per fold
  - predicted_bracket_{year}.csv — per-matchup predictions for 2022/23/24
  - formula_backtest_summary.csv — aggregate scores across holdout years
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import PROCESSED_DIR, EXTERNAL_DIR, ESPN_ROUND_POINTS, SEED_PAIRINGS, REGIONS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

TRAIN_YEARS  = list(range(2013, 2022))   # 2013–2021 inclusive, excl. 2020
HOLDOUT_YEARS = [2022, 2023, 2024]       # predict these, never seen in training

FEATURES = [
    # 2-feature set validated by SHAP temporal CV (8 folds, 2016-2024).
    #
    # WAB was investigated as a 3rd feature to fix Houston-over-Kansas 2022
    # (Kansas WAB=10.4 vs Houston WAB=6.2 despite similar TQS). However TQS
    # and WAB are r=0.93 correlated — L2 regression assigns WAB a *negative*
    # weight when both are present (collinearity sign flip), so adding WAB
    # does not flip Kansas over Houston in any C configuration tested.
    #
    # The likely fix is tournament experience (prior games/minutes), which
    # is orthogonal to efficiency and gave Kansas a 5.5x advantage over
    # Houston (61 games vs 11 games before 2022). See feature/tourney-experience.
    "TRUE_QUALITY_SCORE",  # efficiency: AdjEM - 0.4*Luck. corr(rounds_won)=0.596
    "SEED_DIVERGENCE",     # underseeded identifier: corr(rounds_won)=0.313
]

# L2 regularization strength. Smaller C = stronger regularization.
# Validated by temporal CV grid search in run_temporal_cv(). At 2 features
# the dataset is well-conditioned so C=1.0 (weak regularization) is fine.
# Grid: [0.01, 0.05, 0.1, 0.25, 0.5, 1.0] — see run_temporal_cv() for results.
C_REGULARIZATION: float = 1.0   # tuned by grid search

# DayNum → round is inferred dynamically per year in load_actual_results()
# using game counts (see _build_daynum_to_round). This constant is kept
# only for the ui/app.py compatibility shim — do not use it for scoring.
DAYNUM_TO_ROUND: dict[int, int] = {}  # populated dynamically per year

SEED_PAIRINGS = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
# REGIONS, ESPN_ROUND_POINTS, SEED_PAIRINGS imported from config
ESPN_POINTS = ESPN_ROUND_POINTS  # local alias used throughout this file


# ── Data loading ─────────────────────────────────────────────────────────────

def load_matchup_data() -> pd.DataFrame:
    """
    Build a symmetric matchup DataFrame from tournament results + candidate features.

    Uses features_candidates.csv (enriched with engineered columns like EFF_RATIO)
    rather than features_coaching.csv so all FEATURES are available.

    For each game (A beats B) creates two rows:
      row1: features(A) - features(B), label=1
      row2: features(B) - features(A), label=0

    Returns:
        DataFrame with FEATURES columns + LABEL + YEAR.
    """
    from src.utils.team_names import build_kaggle_to_cbb_map

    feats_df = load_features()

    results_path = EXTERNAL_DIR / "kaggle" / "MNCAATourneyDetailedResults.csv"
    teams_path   = EXTERNAL_DIR / "kaggle" / "MTeams.csv"

    results = pd.read_csv(results_path).rename(columns={"Season": "YEAR"})
    teams   = pd.read_csv(teams_path)
    kaggle_to_cbb = build_kaggle_to_cbb_map(feats_df, teams)

    results["WTEAM"] = results["WTeamID"].map(kaggle_to_cbb)
    results["LTEAM"] = results["LTeamID"].map(kaggle_to_cbb)

    before = len(results)
    results = results.dropna(subset=["WTEAM", "LTEAM"])
    dropped = before - len(results)
    if dropped:
        log.warning(f"Dropped {dropped} games with unmapped team names")

    results = results[results["YEAR"].between(2013, 2024) & (results["YEAR"] != 2020)]

    feat_cols = ["YEAR", "TEAM"] + FEATURES
    feat = feats_df[feat_cols].copy()

    # Build feature lookup: (team, year) -> feature vector
    feat_lookup: dict[tuple[str, int], np.ndarray] = {}
    for _, row in feat.iterrows():
        key = (row["TEAM"], int(row["YEAR"]))
        vals = row[FEATURES].values
        if not np.any(np.isnan(vals.astype(float))):
            feat_lookup[key] = vals

    records = []
    skipped = 0
    for _, row in results.iterrows():
        year = int(row["YEAR"])
        w_key = (row["WTEAM"], year)
        l_key = (row["LTEAM"], year)
        w_feats = feat_lookup.get(w_key)
        l_feats = feat_lookup.get(l_key)
        if w_feats is None or l_feats is None:
            skipped += 1
            continue
        diff = w_feats - l_feats
        records.append({**dict(zip(FEATURES, diff)),  "LABEL": 1, "YEAR": year})
        records.append({**dict(zip(FEATURES, -diff)), "LABEL": 0, "YEAR": year})

    if skipped:
        log.warning(f"Skipped {skipped} games missing features (missing in features_candidates.csv)")

    df = pd.DataFrame(records)
    log.info(f"Matchup data: {len(df)//2} games, {df['YEAR'].nunique()} seasons")
    return df


def load_features() -> pd.DataFrame:
    """
    Load full feature matrix.

    Prefers features_candidates.csv (engineered columns like EFF_RATIO) if available.
    Falls back to features_coaching.csv for the core features (TQS, SEED_DIVERGENCE, etc.).
    """
    for name in ("features_candidates.csv", "features_coaching.csv"):
        path = PROCESSED_DIR / name
        if path.exists():
            if name != "features_coaching.csv":
                log.info(f"Loaded features from {name}")
            return pd.read_csv(path)
    raise FileNotFoundError(
        f"No feature file found in {PROCESSED_DIR}. "
        "Run src/features/efficiency.py to generate features_coaching.csv."
    )


def load_actual_results(year: int) -> dict[str, int]:
    """
    Return {team_name: furthest_round_reached} for a given tournament year.

    Uses Kaggle tournament results + team name mapping.
    Round 6 = champion, 0 = lost in R64 without winning.
    """
    from src.utils.team_names import build_kaggle_to_cbb_map, build_daynum_to_round

    rpath = EXTERNAL_DIR / "kaggle" / "MNCAATourneyDetailedResults.csv"
    tpath = EXTERNAL_DIR / "kaggle" / "MTeams.csv"
    if not rpath.exists() or not tpath.exists():
        return {}

    feats      = load_features()
    teams_df   = pd.read_csv(tpath)
    results_df = pd.read_csv(rpath)
    kaggle_to_cbb = build_kaggle_to_cbb_map(feats, teams_df)

    yr = results_df[results_df["Season"] == year]
    if yr.empty:
        return {}

    daynum_to_round = build_daynum_to_round(yr)

    team_rounds: dict[str, int] = {}
    for _, row in yr.iterrows():
        rnd = daynum_to_round.get(int(row["DayNum"]))
        if rnd is None:
            continue
        w = kaggle_to_cbb.get(int(row["WTeamID"]))
        l = kaggle_to_cbb.get(int(row["LTeamID"]))
        if w:
            team_rounds[w] = max(team_rounds.get(w, 0), rnd)
        if l:
            team_rounds[l] = max(team_rounds.get(l, 0), rnd - 1)
    return {t: r for t, r in team_rounds.items() if r >= 0}


# ── Model training ────────────────────────────────────────────────────────────

def fit_model(
    df: pd.DataFrame,
    c: float = C_REGULARIZATION,
) -> tuple[LogisticRegression, StandardScaler, np.ndarray]:
    """
    Fit L2-regularized logistic regression (Ridge) on matchup data.

    L2 regularization is critical for the 6-feature set: ADJOE and TQS are
    correlated (r≈0.83), as are ADJDE and TQS (r≈0.71). Without regularization,
    LR assigns huge opposing weights to correlated features, which destabilize
    predictions on unseen data. L2 shrinks all weights toward zero jointly,
    distributing the signal across correlated features instead of amplifying one.

    All features are StandardScaled before fitting so the L2 penalty is applied
    equally (L2 penalizes raw coefficient magnitude — unscaled features with
    different ranges would receive unequal effective penalties).

    Args:
        df: Matchup DataFrame with FEATURES + LABEL columns.
        c:  Inverse regularization strength. Smaller = stronger L2 penalty.
            Default is C_REGULARIZATION, tuned by run_temporal_cv().

    Returns:
        (fitted_model, fitted_scaler, raw_coefs_in_original_feature_space)
    """
    clean = df.dropna(subset=FEATURES + ["LABEL"])
    X = clean[FEATURES].values
    y = clean["LABEL"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        # L2 penalty is the default in sklearn; explicitly setting penalty="l2"
        # triggers a deprecation warning in sklearn 1.8+. Leave it as default.
        solver="lbfgs",
        C=c,
        max_iter=2000,
        random_state=42,
    )
    model.fit(X_scaled, y)

    # Un-standardize coefficients so they apply to raw feature values.
    # Derivation: model predicts on scaled input x̃ = (x - μ) / σ.
    # score = Σ w̃_i * x̃_i = Σ (w̃_i / σ_i) * x_i + const
    # So raw_coef[i] = w̃_i / σ_i represents the per-unit-of-raw-feature impact.
    raw_coefs = model.coef_[0] / scaler.scale_

    return model, scaler, raw_coefs


def predict_prob(a_feats: np.ndarray, b_feats: np.ndarray,
                 model: LogisticRegression, scaler: StandardScaler) -> float:
    """
    P(A beats B) from the fitted model.

    Args:
        a_feats: Feature vector for team A (raw, unstandardized).
        b_feats: Feature vector for team B (raw, unstandardized).

    Returns:
        Float probability in (0, 1).
    """
    diff = (a_feats - b_feats).reshape(1, -1)
    X_scaled = scaler.transform(diff)
    return float(model.predict_proba(X_scaled)[0, 1])


# ── Temporal cross-validation ─────────────────────────────────────────────────

def run_temporal_cv(
    df: pd.DataFrame,
    c_grid: list[float] | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Temporal CV with C grid search: for each year in TRAIN_YEARS (min 3 prior
    seasons), train on all PRIOR years only, test on that year.

    Runs the full temporal CV for each C value in c_grid, selects the C that
    minimizes mean log loss across all folds, and returns per-fold results for
    that best C.

    Why grid search over C here instead of cross-validating C inside each fold?
    Because with only 8 training folds total, nested CV would leave too few
    samples per inner fold to be stable. Grid-searching over the outer folds
    is a reasonable approximation given our data constraints.

    Args:
        df:     Matchup DataFrame (FEATURES + LABEL + YEAR).
        c_grid: L2 C values to evaluate. Defaults to 6 values spanning 2 orders
                of magnitude — sufficient to find the right regime.

    Returns:
        Tuple of (per-fold metrics DataFrame for best C, best C value).
        DataFrame columns: year, n_train, n_test, accuracy, log_loss, C.
    """
    if c_grid is None:
        c_grid = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

    train_years = sorted(t for t in TRAIN_YEARS if t != 2020)
    best_c = c_grid[0]
    best_mean_ll = float("inf")
    all_results: dict[float, list[dict]] = {}

    for c in c_grid:
        records = []
        for year in train_years:
            prior = [y for y in train_years if y < year]
            if len(prior) < 3:
                continue

            train_mask = df["YEAR"].isin(prior)
            test_mask  = df["YEAR"] == year

            train_sub = df[train_mask].dropna(subset=FEATURES + ["LABEL"])
            test_sub  = df[test_mask].dropna(subset=FEATURES + ["LABEL"])

            if len(train_sub) < 50 or len(test_sub) == 0:
                continue

            model, scaler, _ = fit_model(train_sub, c=c)
            X_test = test_sub[FEATURES].values
            y_test = test_sub["LABEL"].values
            probs  = model.predict_proba(scaler.transform(X_test))[:, 1]
            preds  = (probs >= 0.5).astype(int)

            records.append({
                "year":     year,
                "n_train":  len(train_sub) // 2,
                "n_test":   len(test_sub) // 2,
                "accuracy": round(accuracy_score(y_test, preds), 4),
                "log_loss": round(log_loss(y_test, np.clip(probs, 1e-7, 1-1e-7)), 4),
                "C":        c,
            })

        if not records:
            continue

        mean_ll = float(np.mean([r["log_loss"] for r in records]))
        log.info(f"  C={c:.3f}: mean_acc={np.mean([r['accuracy'] for r in records]):.4f}  "
                 f"mean_ll={mean_ll:.4f}")
        all_results[c] = records

        if mean_ll < best_mean_ll:
            best_mean_ll = mean_ll
            best_c = c

    log.info(f"Best C from grid search: {best_c} (mean log loss {best_mean_ll:.4f})")

    # Per-fold detail for the best C
    best_records = all_results.get(best_c, [])
    for r in best_records:
        log.info(f"  Temporal CV {r['year']}: acc={r['accuracy']:.4f}  "
                 f"ll={r['log_loss']:.4f}  (C={r['C']}, n_train={r['n_train']})")

    return pd.DataFrame(best_records), best_c


# ── Formula extraction ────────────────────────────────────────────────────────

def extract_formula(model: LogisticRegression, scaler: StandardScaler,
                    raw_coefs: np.ndarray) -> pd.DataFrame:
    """
    Extract the explicit mathematical formula from the fitted model.

    The model is:
        P(A beats B) = σ( Σ_i  w_i * (feat_i(A) - feat_i(B)) )

    Which is equivalent to a linear TEAM SCORE:
        score(T) = Σ_i  w_i * feat_i(T)
        P(A beats B) = σ( score(A) - score(B) )

    Args:
        model:     Fitted LogisticRegression.
        scaler:    Fitted StandardScaler.
        raw_coefs: Un-standardized coefficients.

    Returns:
        DataFrame with feature, weight (raw), weight (normalized), and interpretation.
    """
    # Normalize weights to sum of abs = 1 for display
    abs_sum = np.abs(raw_coefs).sum()
    norm_weights = raw_coefs / abs_sum if abs_sum > 0 else raw_coefs

    interpretations = {
        "TRUE_QUALITY_SCORE": "higher = stronger team (AdjEM - 0.4*Luck)",
        "WAB":                "wins above bubble — resume quality / schedule difficulty",
        "SEED_DIVERGENCE":    "positive = underseeded (KenPom ranks them better than seed)",
        "QMS":                "quality momentum — weighted wins vs top-25/50/100 teams",
        "COACH_PREMIUM":      "career tournament wins above seed expectation (Tom Izzo signal)",
        "ADJOE":              "adjusted offensive efficiency (pts per 100 possessions)",
        "ADJDE":              "adjusted defensive efficiency (lower = better defense)",
    }

    df = pd.DataFrame({
        "feature":         FEATURES,
        "raw_weight":      raw_coefs.round(6),
        "norm_weight":     norm_weights.round(4),
        "interpretation":  [interpretations[f] for f in FEATURES],
        "feature_mean":    scaler.mean_.round(4),
        "feature_std":     scaler.scale_.round(4),
    }).sort_values("norm_weight", key=abs, ascending=False)

    return df


# ── Bracket simulation for a given year ──────────────────────────────────────

def simulate_bracket(year: int, model: LogisticRegression,
                     scaler: StandardScaler) -> dict:
    """
    Simulate the full tournament bracket for a holdout year.

    Uses the formula: pick the team with higher P(win) at every game node.
    Tracks every matchup with predicted winner, win probability, actual winner,
    and whether the prediction was correct.

    Args:
        year:   Tournament year to simulate.
        model:  Fitted logistic regression.
        scaler: Fitted StandardScaler.

    Returns:
        Dict with keys: champion, matchups (list), team_rounds (dict),
        actual (dict), espn_score (int|None).
    """
    feats_df = load_features()
    # Load tournament teams — SEED is needed for bracket structure, not as a model feature
    tourney_base = feats_df[
        (feats_df["YEAR"] == year) & feats_df["SEED"].notna()
    ][["TEAM", "SEED"] + FEATURES].copy()
    tourney_base["SEED"] = tourney_base["SEED"].astype(int)
    tourney = tourney_base.dropna(subset=FEATURES)

    actual = load_actual_results(year)
    feat_lookup = {row["TEAM"]: row[FEATURES].values for _, row in tourney.iterrows()}
    seed_lookup  = dict(zip(tourney["TEAM"], tourney["SEED"]))

    from collections import defaultdict
    by_seed: dict[int, list[str]] = defaultdict(list)
    for _, row in tourney.iterrows():
        by_seed[row["SEED"]].append(row["TEAM"])

    # First Four: if >4 teams share a seed, keep only the 4 with highest avg win prob
    def resolve_first_four(teams: list[str]) -> list[str]:
        scores = {}
        for t in teams:
            if t not in feat_lookup:
                scores[t] = 0.0
                continue
            probs = []
            for o in teams:
                if o != t and o in feat_lookup:
                    probs.append(predict_prob(feat_lookup[t], feat_lookup[o], model, scaler))
            scores[t] = np.mean(probs) if probs else 0.5
        return sorted(teams, key=lambda t: scores[t], reverse=True)[:4]

    region_teams: dict[str, dict[int, str]] = {r: {} for r in REGIONS}
    for seed, teams in sorted(by_seed.items()):
        main = resolve_first_four(teams) if len(teams) > 4 else teams[:4]
        for i, t in enumerate(main):
            region_teams[REGIONS[i]][seed] = t

    def game(ta: str, tb: str, rnd: int) -> dict:
        """Simulate one game, return matchup record."""
        if ta in feat_lookup and tb in feat_lookup:
            p = predict_prob(feat_lookup[ta], feat_lookup[tb], model, scaler)
        else:
            p = 0.5
        winner = ta if p >= 0.5 else tb
        loser  = tb if winner == ta else ta

        act_winner = None
        if actual:
            if actual.get(ta, -1) >= rnd:
                act_winner = ta
            elif actual.get(tb, -1) >= rnd:
                act_winner = tb

        correct = (act_winner is not None) and (winner == act_winner)

        return {
            "round":            rnd,
            "team_a":           ta,
            "team_b":           tb,
            "seed_a":           seed_lookup.get(ta),
            "seed_b":           seed_lookup.get(tb),
            "prob_a":           round(p, 4),
            "prob_b":           round(1 - p, 4),
            "predicted_winner": winner,
            "actual_winner":    act_winner,
            "correct":          correct if act_winner is not None else None,
            "region":           None,  # filled in below
        }

    all_matchups = []
    region_champs = {}

    for region in REGIONS:
        rt = region_teams[region]
        current_round_teams = []

        # R64
        for high, low in SEED_PAIRINGS:
            ta = rt.get(high, f"Seed {high}")
            tb = rt.get(low,  f"Seed {low}")
            m = game(ta, tb, 1)
            m["region"] = region
            all_matchups.append(m)
            current_round_teams.append(m["predicted_winner"])

        # R32, S16, E8
        for rnd in [2, 3, 4]:
            next_teams = []
            for i in range(0, len(current_round_teams), 2):
                if i + 1 >= len(current_round_teams):
                    break
                ta, tb = current_round_teams[i], current_round_teams[i+1]
                m = game(ta, tb, rnd)
                m["region"] = region
                all_matchups.append(m)
                next_teams.append(m["predicted_winner"])
            current_round_teams = next_teams

        region_champs[region] = current_round_teams[0] if current_round_teams else None

    # Final Four
    f4_pairs = [(REGIONS[0], REGIONS[1]), (REGIONS[2], REGIONS[3])]
    f4_winners = []
    for ra, rb in f4_pairs:
        ta = region_champs[ra]
        tb = region_champs[rb]
        m = game(ta, tb, 5)
        m["region"] = "Final Four"
        all_matchups.append(m)
        f4_winners.append(m["predicted_winner"])

    # Championship
    m = game(f4_winners[0], f4_winners[1], 6)
    m["region"] = "Championship"
    all_matchups.append(m)
    champion = m["predicted_winner"]

    # Build team → furthest predicted round reached
    team_rounds: dict[str, int] = {}
    for m in all_matchups:
        w = m["predicted_winner"]
        r = m["round"]
        if w:
            team_rounds[w] = max(team_rounds.get(w, 0), r)
        loser = m["team_b"] if m["predicted_winner"] == m["team_a"] else m["team_a"]
        team_rounds[loser] = max(team_rounds.get(loser, 0), r - 1)

    # ESPN score
    espn_score = None
    if actual:
        espn_score = 0
        for team, pred_rnd in team_rounds.items():
            for r in range(1, pred_rnd + 1):
                if actual.get(team, 0) >= r:
                    espn_score += ESPN_POINTS.get(r, 0)

    return {
        "year":        year,
        "champion":    champion,
        "matchups":    all_matchups,
        "team_rounds": team_rounds,
        "actual":      actual,
        "espn_score":  espn_score,
    }


def compute_round_accuracy(matchups: list[dict]) -> dict[int, float]:
    """Compute pick accuracy per round for matchups that have actual results."""
    from collections import defaultdict
    correct_by_round: dict[int, int] = defaultdict(int)
    total_by_round:   dict[int, int] = defaultdict(int)
    for m in matchups:
        if m["correct"] is not None:
            r = m["round"]
            total_by_round[r] += 1
            if m["correct"]:
                correct_by_round[r] += 1
    return {
        r: round(correct_by_round[r] / total_by_round[r], 4)
        for r in sorted(total_by_round)
        if total_by_round[r] > 0
    }


# ── Win probability matrix (feeds directly into simulator + optimizer) ────────

def build_win_prob_matrix_from_formula(
    year: int,
    model: LogisticRegression,
    scaler: StandardScaler,
    features_path: Path = PROCESSED_DIR / "features_coaching.csv",
) -> pd.DataFrame:
    """
    Build a square win probability matrix for the simulator using the formula model.

    This is the wiring point between formula_model.py (Layer 1) and
    simulator.py / optimizer.py (Layers 2-3).  The simulator expects:
      - A square pd.DataFrame indexed by team name
      - matrix.loc[team_a, team_b] = P(team_a beats team_b)
      - Values in [0, 1]; diagonal is 0.5

    Team names must match whatever is in _BRACKET_2025_SEEDS in optimizer.py
    and in features_coaching.csv TEAM column — both use the same CBB names,
    so no name bridging is needed here.

    Uses vectorized batch predict_proba: builds all n*(n-1) pairwise feature
    difference vectors at once, runs a single scaler.transform + predict_proba
    call.  ~300x faster than nested loops for a 68-team field.

    Args:
        year:          Tournament year to build the matrix for.
        model:         Fitted LogisticRegression (from fit_model()).
        scaler:        Fitted StandardScaler (from fit_model()).
        features_path: Path to features_coaching.csv.

    Returns:
        Square DataFrame (n_teams × n_teams) with float win probabilities.
        Returns an empty DataFrame if fewer than 2 tournament teams are found.
    """
    features = pd.read_csv(features_path)
    tourney = features[
        (features["YEAR"] == year) & features["SEED"].notna()
    ][["TEAM"] + FEATURES].copy()

    # Drop rows missing any feature — model can't predict without them
    before = len(tourney)
    tourney = tourney.dropna(subset=FEATURES)
    if before - len(tourney) > 0:
        log.warning(
            f"Dropped {before - len(tourney)} tournament teams for {year} "
            f"with missing features — they will not appear in the matrix"
        )

    teams = tourney["TEAM"].tolist()
    n = len(teams)

    if n < 2:
        log.warning(f"Only {n} teams with full features for {year} — cannot build matrix")
        return pd.DataFrame()

    # Pre-build a dict for O(1) feature lookups
    feat_dict: dict[str, np.ndarray] = {
        row["TEAM"]: row[FEATURES].values.astype(float)
        for _, row in tourney.iterrows()
    }

    # All off-diagonal pairs as a flat list of (i, j) index tuples
    pair_idx = [(i, j) for i in range(n) for j in range(n) if i != j]

    # Stack all feature differences into a single matrix: shape (n*(n-1), n_features)
    # Then one scaler.transform + one predict_proba call instead of n*(n-1) individual calls.
    diff_matrix = np.vstack([
        feat_dict[teams[i]] - feat_dict[teams[j]]
        for i, j in pair_idx
    ])
    probs_flat = model.predict_proba(scaler.transform(diff_matrix))[:, 1]

    # Fill the square DataFrame
    matrix = pd.DataFrame(0.5, index=teams, columns=teams, dtype=float)
    for k, (i, j) in enumerate(pair_idx):
        matrix.iloc[i, j] = round(float(probs_flat[k]), 4)

    log.info(
        f"Win probability matrix built: {n}×{n} teams for {year} "
        f"(features: {FEATURES})"
    )
    return matrix


# ── Full integrated pipeline (formula model → simulator → optimizer) ──────────

def run_full_pipeline(
    target_year: int = 2025,
    train_years: list[int] | None = None,
    n_sims: int = 10_000,
    sim_seed: int = 42,
    save_matrix: bool = True,
) -> dict:
    """
    Full end-to-end pipeline: train formula model → build win prob matrix →
    run Monte Carlo simulation → build chalk/medium/chaos brackets.

    This is the wiring function that connects formula_model.py (Layer 1)
    directly to simulator.py (Layer 2) and optimizer.py (Layer 3).

    Previously these pipelines were disconnected:
      - formula_model.py produced deterministic bracket CSVs
      - simulator.py + optimizer.py read win_prob_matrix from win_probability.py
    Now a single call produces all three brackets using the formula model's
    calibrated probabilities.

    The L2 regularized model is trained on all available seasons before
    target_year (temporal constraint strictly enforced — no future data),
    then the win probability matrix is fed directly into run_simulations().

    Args:
        target_year: Year to generate brackets for (default 2025).
        train_years: Explicit list of years to train on. Defaults to all
                     years in features_coaching.csv strictly before target_year,
                     excluding 2020 (no tournament).
        n_sims:      Monte Carlo simulation count (default 10,000).
        sim_seed:    Random seed for the simulator.
        save_matrix: Whether to save the win prob matrix to
                     data/processed/win_prob_matrix_{year}.csv (default True).

    Returns:
        Dict with keys:
          model         — fitted LogisticRegression
          scaler        — fitted StandardScaler
          raw_coefs     — un-standardized coefficients
          formula_df    — feature weight DataFrame
          win_prob_matrix — square DataFrame (n_teams × n_teams)
          sim_results   — full output from run_simulations()
          brackets      — {"chalk": ..., "medium": ..., "chaos": ...}
          p_reach       — P(reach) DataFrame from build_p_reach()
          best_c        — C value selected by temporal CV grid search
    """
    from src.models.simulator import run_simulations, build_bracket_from_seeds
    from src.models.optimizer import build_bracket, build_p_reach, load_seed_divergence, _build_bracket_structure_2025

    # ── 1. Load matchup data ───────────────────────────────────────────────────
    log.info(f"run_full_pipeline: target_year={target_year}, n_sims={n_sims:,}")
    df = load_matchup_data()
    log.info(f"  Matchup data: {len(df)//2} games across {df['YEAR'].nunique()} seasons")

    # ── 2. Select training years (strict temporal — no future data) ────────────
    all_years_in_data = sorted(df["YEAR"].unique())
    if train_years is None:
        train_years = [y for y in all_years_in_data if y < target_year and y != 2020]

    if not train_years:
        raise ValueError(
            f"No training data available before {target_year}. "
            f"Years in dataset: {all_years_in_data}"
        )

    train_df = df[df["YEAR"].isin(train_years)].copy()
    log.info(
        f"  Training on {len(train_years)} seasons: {train_years[0]}–{train_years[-1]} "
        f"({len(train_df)//2} games)"
    )

    # ── 3. Temporal CV + C grid search (validates model stability) ────────────
    log.info("  Running temporal CV with C grid search...")
    cv_results, best_c = run_temporal_cv(train_df)
    if len(cv_results) > 0:
        log.info(
            f"  CV result: mean_acc={cv_results['accuracy'].mean():.4f}, "
            f"mean_ll={cv_results['log_loss'].mean():.4f}, best_C={best_c}"
        )

    # ── 4. Fit final model on all training years ───────────────────────────────
    log.info(f"  Fitting final model (C={best_c}) on all {len(train_years)} training seasons...")
    model, scaler, raw_coefs = fit_model(train_df, c=best_c)

    formula_df = extract_formula(model, scaler, raw_coefs)
    formula_df.to_csv(PROCESSED_DIR / "formula_weights.csv", index=False)
    log.info("  Formula weights saved.")

    # ── 5. Build win probability matrix for target year ────────────────────────
    log.info(f"  Building win probability matrix for {target_year}...")
    win_prob_matrix = build_win_prob_matrix_from_formula(target_year, model, scaler)

    if win_prob_matrix.empty:
        raise RuntimeError(
            f"Win probability matrix is empty for {target_year}. "
            "Check that features_coaching.csv has SEED data for this year."
        )

    if save_matrix:
        out_path = PROCESSED_DIR / f"win_prob_matrix_{target_year}.csv"
        win_prob_matrix.to_csv(out_path)
        log.info(f"  Matrix saved to {out_path}")

    # ── 6. Build bracket structure ─────────────────────────────────────────────
    # Try Kaggle seeds file first (has real seedings for historical years).
    # Fall back to hardcoded 2025 Selection Sunday structure.
    seeds_path = EXTERNAL_DIR / "kaggle" / "MNCAATourneySeeds.csv"
    teams_path = EXTERNAL_DIR / "kaggle" / "MTeams.csv"

    bracket_structure = None
    if seeds_path.exists() and teams_path.exists():
        seeds_df = pd.read_csv(seeds_path)
        teams_df = pd.read_csv(teams_path)
        yr_seeds = seeds_df[seeds_df["Season"] == target_year]
        if not yr_seeds.empty:
            bracket_structure = build_bracket_from_seeds(seeds_df, teams_df, target_year)
            log.info(f"  Bracket structure built from Kaggle seeds for {target_year}.")

    if bracket_structure is None:
        if target_year == 2025:
            bracket_structure = _build_bracket_structure_2025()
            log.info("  Using hardcoded 2025 Selection Sunday bracket structure.")
        else:
            raise RuntimeError(
                f"Cannot build bracket structure for {target_year}: "
                "Kaggle seeds file not found or has no data for this year."
            )

    # ── 7. Remap matrix team names to match bracket structure ──────────────────
    # The bracket structure uses Kaggle team names (from MTeams.csv or the
    # hardcoded _BRACKET_2025_SEEDS dict). The win prob matrix uses CBB names
    # from features_coaching.csv. For 2025 these are the same strings, but
    # we verify coverage and warn on mismatches rather than silently failing.
    bracket_teams: set[str] = set()
    for matchups in bracket_structure["regions"].values():
        for ta, tb in matchups:
            bracket_teams.update([ta, tb])
    for ta, tb, *_ in bracket_structure.get("first_four", []):
        bracket_teams.update([ta, tb])

    matrix_teams = set(win_prob_matrix.index)
    missing_from_matrix = bracket_teams - matrix_teams - {t for t in bracket_teams if t.startswith("FF_")}
    if missing_from_matrix:
        log.warning(
            f"  {len(missing_from_matrix)} bracket teams missing from win prob matrix "
            f"(will use 50/50 fallback in simulator): {sorted(missing_from_matrix)}"
        )

    # ── 8. Run Monte Carlo simulation ─────────────────────────────────────────
    log.info(f"  Running {n_sims:,} Monte Carlo simulations...")
    sim_results = run_simulations(
        win_prob_matrix, bracket_structure, n_sims=n_sims, seed=sim_seed
    )
    log.info(
        f"  Simulation complete. Top champions: "
        + ", ".join(
            f"{row['TEAM']} ({row['FREQUENCY_PCT']:.1f}%)"
            for _, row in __import__("src.models.simulator", fromlist=["top_champions"])
            .top_champions(sim_results["champions"], n=3).iterrows()
        )
    )

    # ── 9. Build P(reach) table and seed divergence ────────────────────────────
    from src.models.optimizer import build_p_reach
    p_reach = build_p_reach(sim_results)
    seed_div = load_seed_divergence(year=target_year)

    # ── 10. Build chalk / medium / chaos brackets ──────────────────────────────
    log.info("  Building chalk / medium / chaos brackets...")
    chalk  = build_bracket(p_reach, seed_div, win_prob_matrix, mode="chalk",  bracket_structure=bracket_structure)
    medium = build_bracket(p_reach, seed_div, win_prob_matrix, mode="medium", bracket_structure=bracket_structure)
    chaos  = build_bracket(p_reach, seed_div, win_prob_matrix, mode="chaos",  bracket_structure=bracket_structure)

    log.info(
        f"  Champions — chalk: {chalk['champion']}, "
        f"medium: {medium['champion']}, chaos: {chaos['champion']}"
    )

    # Save brackets CSV
    from src.models.optimizer import save_brackets, print_bracket_comparison, print_expected_scores
    brackets_list = [chalk, medium, chaos]
    save_brackets(brackets_list, year=target_year)

    return {
        "model":            model,
        "scaler":           scaler,
        "raw_coefs":        raw_coefs,
        "formula_df":       formula_df,
        "win_prob_matrix":  win_prob_matrix,
        "sim_results":      sim_results,
        "brackets":         {"chalk": chalk, "medium": medium, "chaos": chaos},
        "p_reach":          p_reach,
        "best_c":           best_c,
        "cv_results":       cv_results,
        "bracket_structure": bracket_structure,
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def main() -> None:
    """
    Full pipeline:
      1. Load data
      2. Temporal CV on train years → validate formula stability
      3. Fit final model on all train years → extract formula weights
      4. Generate bracket predictions for holdout years (2022, 2023, 2024)
      5. Save all outputs
    """
    log.info("Loading matchup data...")
    df = load_matchup_data()
    log.info(f"  {len(df)//2} tournament games ({df['YEAR'].nunique()} seasons)")

    # ── Step 1: Temporal CV + C grid search on training years ─────────────────
    log.info("\nRunning temporal CV with L2 C grid search on training years (2013–2021)...")
    log.info(f"  Features: {FEATURES}")
    train_df = df[df["YEAR"].isin(TRAIN_YEARS) & (df["YEAR"] != 2020)]
    cv_results, best_c = run_temporal_cv(train_df)

    print("\n" + "=" * 65)
    print(f"TEMPORAL CV RESULTS — best C={best_c} (L2 Ridge, 6 features)")
    print("=" * 65)
    print(cv_results.to_string(index=False))
    if len(cv_results) > 0:
        print(f"\n  Mean accuracy:  {cv_results['accuracy'].mean():.4f}")
        print(f"  Mean log loss:  {cv_results['log_loss'].mean():.4f}")
        print(f"  Accuracy range: {cv_results['accuracy'].min():.4f} – {cv_results['accuracy'].max():.4f}")
        print(f"  Best C:         {best_c}")

    cv_results.to_csv(PROCESSED_DIR / "formula_cv_results.csv", index=False)
    log.info(f"CV results saved to {PROCESSED_DIR/'formula_cv_results.csv'}")

    # ── Step 2: Fit final model on all training years using best C ─────────────
    log.info(f"\nFitting final model on all training data (2013–2021, excl. 2020) with C={best_c}...")
    final_model, final_scaler, raw_coefs = fit_model(train_df, c=best_c)

    # ── Step 3: Extract and display the formula ───────────────────────────────
    formula_df = extract_formula(final_model, final_scaler, raw_coefs)

    print("\n" + "=" * 75)
    print("THE MODEL FORMULA  (learned from 2013–2021 tournament data)")
    print("=" * 75)
    print()
    print("  P(A beats B) = σ( SCORE(A) - SCORE(B) )  where σ is sigmoid")
    print()
    print("  SCORE(T) = " + " +\n             ".join(
        f"({w:+.6f}) × {f}"
        for f, w in zip(formula_df["feature"], formula_df["raw_weight"])
    ))
    print()
    print("  Feature weights (sorted by importance):")
    print(f"  {'Feature':<22} {'Raw Weight':>12}  {'Normalized':>10}  Interpretation")
    print("  " + "-" * 70)
    for _, row in formula_df.iterrows():
        print(f"  {row['feature']:<22} {row['raw_weight']:>12.6f}  {row['norm_weight']:>10.4f}  {row['interpretation']}")

    formula_df.to_csv(PROCESSED_DIR / "formula_weights.csv", index=False)
    log.info(f"\nFormula weights saved to {PROCESSED_DIR/'formula_weights.csv'}")

    # ── Step 4: Generate bracket predictions for holdout years ────────────────
    print("\n" + "=" * 65)
    print("HOLDOUT YEAR PREDICTIONS (never seen during training)")
    print("=" * 65)

    summary_records = []

    for year in HOLDOUT_YEARS:
        log.info(f"\nSimulating bracket for {year}...")
        result = simulate_bracket(year, final_model, final_scaler)

        matchups = result["matchups"]
        actual   = result["actual"]
        champion = result["champion"]
        espn     = result["espn_score"]

        # Round accuracy
        round_acc = compute_round_accuracy(matchups)
        ROUND_NAMES = {1:"R64", 2:"R32", 3:"S16", 4:"E8", 5:"F4", 6:"Champ"}

        actual_champ = None
        if actual:
            actual_champ = max(actual, key=lambda t: actual[t]) if actual else None
            for t, r in actual.items():
                if r == 6:
                    actual_champ = t
                    break

        champ_correct = (actual_champ is not None) and (champion == actual_champ)

        print(f"\n  Year {year}:")
        print(f"    Predicted champion : {champion}")
        print(f"    Actual champion    : {actual_champ or 'N/A'}")
        print(f"    Champion correct   : {'✓ YES' if champ_correct else '✗ NO'}")
        print(f"    ESPN score         : {espn if espn is not None else 'N/A'}")
        for rnd in sorted(round_acc):
            n_games = sum(1 for m in matchups if m["round"] == rnd and m["correct"] is not None)
            n_correct = sum(1 for m in matchups if m["round"] == rnd and m["correct"])
            print(f"    {ROUND_NAMES.get(rnd,'R'+str(rnd))} accuracy    : "
                  f"{round_acc[rnd]:.1%} ({n_correct}/{n_games})")

        # Save per-matchup CSV
        matchup_rows = []
        for m in matchups:
            matchup_rows.append({
                "year":             year,
                "region":           m["region"],
                "round":            m["round"],
                "round_name":       ROUND_NAMES.get(m["round"], f"R{m['round']}"),
                "team_a":           m["team_a"],
                "team_b":           m["team_b"],
                "seed_a":           m["seed_a"],
                "seed_b":           m["seed_b"],
                "prob_a":           m["prob_a"],
                "prob_b":           m["prob_b"],
                "predicted_winner": m["predicted_winner"],
                "actual_winner":    m["actual_winner"],
                "correct":          m["correct"],
            })
        matchup_df_out = pd.DataFrame(matchup_rows)
        out_path = PROCESSED_DIR / f"predicted_bracket_{year}.csv"
        matchup_df_out.to_csv(out_path, index=False)
        log.info(f"  Saved {len(matchup_df_out)} matchups to {out_path}")

        rec = {
            "year":               year,
            "predicted_champion": champion,
            "actual_champion":    actual_champ,
            "champion_correct":   champ_correct,
            "espn_score":         espn,
        }
        for rnd, acc in round_acc.items():
            rec[f"{ROUND_NAMES.get(rnd,'r'+str(rnd))}_acc"] = acc
        summary_records.append(rec)

    # ── Step 5: Summary ───────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(PROCESSED_DIR / "formula_backtest_summary.csv", index=False)

    print("\n" + "=" * 65)
    print("AGGREGATE BACKTEST SUMMARY (2022–2024)")
    print("=" * 65)
    print(summary_df.to_string(index=False))

    valid_scores = [r["espn_score"] for r in summary_records if r["espn_score"] is not None]
    if valid_scores:
        max_espn = sum(ESPN_POINTS[r] * (2**(6-r)) for r in range(1,7))
        print(f"\n  Mean ESPN score:    {np.mean(valid_scores):.1f} / {max_espn}")
        print(f"  Champion accuracy:  {sum(r['champion_correct'] for r in summary_records)}/{len(summary_records)}")

    print(f"\n  Files saved to {PROCESSED_DIR}/")
    print("  - formula_weights.csv")
    print("  - formula_cv_results.csv")
    for y in HOLDOUT_YEARS:
        print(f"  - predicted_bracket_{y}.csv")
    print("  - formula_backtest_summary.csv")

    # ── Step 6: 2025 live prediction via full integrated pipeline ─────────────
    # Train on all data through 2024, wire into simulator + optimizer.
    # This is the actual competition bracket — not a holdout validation.
    print("\n" + "=" * 65)
    print("2025 LIVE PREDICTION — Full Pipeline (Formula → Sim → Optimizer)")
    print("=" * 65)
    log.info("Running full integrated pipeline for 2025...")

    try:
        results_2025 = run_full_pipeline(
            target_year=2025,
            n_sims=10_000,
            sim_seed=42,
            save_matrix=True,
        )

        from src.models.optimizer import (
            print_bracket_comparison,
            print_expected_scores,
            top_champions,
        )
        from src.models.simulator import top_champions as sim_top_champions

        print("\nTop 10 Champions (10,000 simulations):")
        print(sim_top_champions(results_2025["sim_results"]["champions"], n=10).to_string(index=False))

        brackets_list = [
            results_2025["brackets"]["chalk"],
            results_2025["brackets"]["medium"],
            results_2025["brackets"]["chaos"],
        ]
        print_bracket_comparison(brackets_list)
        print_expected_scores(brackets_list, results_2025["p_reach"])

        print(f"\n  Chalk champion:  {results_2025['brackets']['chalk']['champion']}")
        print(f"  Medium champion: {results_2025['brackets']['medium']['champion']}")
        print(f"  Chaos champion:  {results_2025['brackets']['chaos']['champion']}")
        print(f"\n  Win prob matrix: {PROCESSED_DIR}/win_prob_matrix_2025.csv")
        print(f"  Three brackets:  {PROCESSED_DIR}/three_brackets_2025.csv")

    except Exception as exc:
        log.error(f"2025 pipeline failed: {exc}")
        raise


if __name__ == "__main__":
    main()
