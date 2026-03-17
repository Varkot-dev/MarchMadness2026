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
    "TRUE_QUALITY_SCORE",  # AdjEM - 0.4*Luck: true team strength (strongest raw signal)
    "SEED_DIVERGENCE",     # actual_seed - KenPom_implied_seed: positive = underseeded upset threat
    "QMS",                 # Quality Momentum Score: recency signal, weighted by opponent strength
    "COACH_PREMIUM",       # Career tournament wins above seed expectation
    "ADJOE",               # Adjusted offensive efficiency (partial independence from TQS)
    "ADJDE",               # Adjusted defensive efficiency (partially independent from TQS)
    # NOTE: SEED, WAB dropped — collinear with TQS at r>0.83, VIF>10 even under L2
    # NOTE: TOR, ORB dropped — noise at 668 samples, collinear with ADJOE
    # L2 regularization (Ridge) handles ADJOE/ADJDE collinearity with TQS without
    # dropping them — it distributes weight across correlated features rather than
    # amplifying one arbitrarily. C is tuned via temporal CV grid search below.
]

# L2 regularization strength. Smaller C = stronger regularization.
# Tuned via temporal CV grid search in run_temporal_cv(); best value stored here
# as the production default after running main(). Start with tight regularization
# given the 668-game dataset — L2 is the only thing keeping ADJOE/ADJDE stable.
# Grid: [0.01, 0.05, 0.1, 0.25, 0.5, 1.0] — see run_temporal_cv() for results.
C_REGULARIZATION: float = 1.0   # production default; tuned by grid search (best log loss 0.4677)

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
    Build a symmetric matchup DataFrame from tournament results + features.

    For each game (A beats B) creates two rows:
      row1: features(A) - features(B), label=1
      row2: features(B) - features(A), label=0

    Returns:
        DataFrame with FEATURES columns + LABEL + YEAR.
    """
    from src.models.win_probability import load_matchup_data as _load
    df = _load()
    # Restrict to years with stable feature coverage
    df = df[df["YEAR"].between(2013, 2024) & (df["YEAR"] != 2020)]
    return df


def load_features() -> pd.DataFrame:
    """Load full feature matrix for all years."""
    return pd.read_csv(PROCESSED_DIR / "features_coaching.csv")


def load_actual_results(year: int) -> dict[str, int]:
    """
    Return {team_name: furthest_round_reached} for a given tournament year.

    Uses Kaggle tournament results + team name mapping.
    Round 6 = champion, 0 = lost in R64 without winning.
    """
    rpath = EXTERNAL_DIR / "kaggle" / "MNCAATourneyDetailedResults.csv"
    tpath = EXTERNAL_DIR / "kaggle" / "MTeams.csv"
    if not rpath.exists() or not tpath.exists():
        return {}

    from src.models.win_probability import _build_kaggle_to_cbb_map
    feats = pd.read_csv(PROCESSED_DIR / "features_coaching.csv")
    teams_df = pd.read_csv(tpath)
    results_df = pd.read_csv(rpath)
    kaggle_to_cbb = _build_kaggle_to_cbb_map(feats, teams_df)

    yr = results_df[results_df["Season"] == year]
    if yr.empty:
        return {}

    # Infer DayNum → round dynamically from game counts so this works across
    # all years including the 2021 bubble. Reuse backtest.py helper.
    from src.evaluation.backtest import _build_daynum_to_round
    daynum_to_round = _build_daynum_to_round(yr)

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


if __name__ == "__main__":
    main()
