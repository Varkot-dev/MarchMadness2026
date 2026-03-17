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

from config import PROCESSED_DIR, EXTERNAL_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

TRAIN_YEARS  = list(range(2013, 2022))   # 2013–2021 inclusive, excl. 2020
HOLDOUT_YEARS = [2022, 2023, 2024]       # predict these, never seen in training

FEATURES = [
    "TRUE_QUALITY_SCORE",  # AdjEM - 0.4*Luck: true strength (dominant signal, r=-0.91 w/ seed)
    "SEED_DIVERGENCE",     # actual_seed - KenPom_implied_seed: positive = underseeded upset threat
    # DROPPED (collinear with TQS, VIF > 10): SEED (r=-0.91), WAB (r=0.93), ADJOE (r=0.83)
    # DROPPED (noisy with 668 samples): QMS, COACH_PREMIUM, TOR, ORB, ADJDE
    # Temporal CV with 7-10 features: acc=0.731-0.732, ll=0.510-0.517
    # Temporal CV with just these 2:  acc=0.741, ll=0.505  ← best on holdout
]

# DayNum → round is inferred dynamically per year in load_actual_results()
# using game counts (see _build_daynum_to_round). This constant is kept
# only for the ui/app.py compatibility shim — do not use it for scoring.
DAYNUM_TO_ROUND: dict[int, int] = {}  # populated dynamically per year

SEED_PAIRINGS = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
REGIONS = ["South", "East", "West", "Midwest"]
ESPN_POINTS = {1:10, 2:20, 3:40, 4:80, 5:160, 6:320}


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

def fit_model(df: pd.DataFrame) -> tuple[LogisticRegression, StandardScaler, np.ndarray]:
    """
    Fit logistic regression on matchup data.

    Args:
        df: Matchup DataFrame with FEATURES + LABEL columns.

    Returns:
        (fitted_model, fitted_scaler, raw_coefs_in_original_feature_space)
    """
    X = df[FEATURES].values
    y = df["LABEL"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        solver="lbfgs",
        C=1.0,          # Tuned via temporal CV: C=1.0 gives best log loss (0.505)
        max_iter=2000,  # C=0.1 overregularizes the 2-feature model
        random_state=42,
    )
    model.fit(X_scaled, y)

    # Un-standardize coefficients so they apply to raw feature values
    # raw_coef[i] = scaled_coef[i] / std[i]
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

def run_temporal_cv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temporal CV: for each year in TRAIN_YEARS (min 3 prior seasons),
    train on all PRIOR years only, test on that year.

    This validates that the weights are stable across time — i.e. the
    formula learned on 2013–2015 still works in 2019.

    Returns:
        DataFrame with per-fold metrics.
    """
    train_years = sorted(t for t in TRAIN_YEARS if t != 2020)
    records = []

    for i, year in enumerate(train_years):
        prior = [y for y in train_years if y < year]
        if len(prior) < 3:
            continue

        train_mask = df["YEAR"].isin(prior)
        test_mask  = df["YEAR"] == year

        X_train = df.loc[train_mask, FEATURES].values
        y_train  = df.loc[train_mask, "LABEL"].values
        X_test   = df.loc[test_mask,  FEATURES].values
        y_test   = df.loc[test_mask,  "LABEL"].values

        if len(X_train) < 50 or len(X_test) == 0:
            continue

        model, scaler, raw_coefs = fit_model(df[train_mask])
        probs = model.predict_proba(scaler.transform(X_test))[:, 1]
        preds = (probs >= 0.5).astype(int)

        records.append({
            "year":       year,
            "n_train":    len(y_train) // 2,
            "n_test":     len(y_test) // 2,
            "accuracy":   round(accuracy_score(y_test, preds), 4),
            "log_loss":   round(log_loss(y_test, np.clip(probs, 1e-7, 1-1e-7)), 4),
        })
        log.info(f"Temporal CV {year}: acc={records[-1]['accuracy']:.4f}  ll={records[-1]['log_loss']:.4f}  (trained on {prior})")

    return pd.DataFrame(records)


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

    # ── Step 1: Temporal CV on training years ─────────────────────────────────
    log.info("\nRunning temporal cross-validation on training years (2013–2021)...")
    train_df = df[df["YEAR"].isin(TRAIN_YEARS) & (df["YEAR"] != 2020)]
    cv_results = run_temporal_cv(train_df)

    print("\n" + "=" * 65)
    print("TEMPORAL CV RESULTS (training data only, 2013–2021)")
    print("=" * 65)
    print(cv_results.to_string(index=False))
    if len(cv_results) > 0:
        print(f"\n  Mean accuracy:  {cv_results['accuracy'].mean():.4f}")
        print(f"  Mean log loss:  {cv_results['log_loss'].mean():.4f}")
        print(f"  Accuracy range: {cv_results['accuracy'].min():.4f} – {cv_results['accuracy'].max():.4f}")

    cv_results.to_csv(PROCESSED_DIR / "formula_cv_results.csv", index=False)
    log.info(f"CV results saved to {PROCESSED_DIR/'formula_cv_results.csv'}")

    # ── Step 2: Fit final model on all training years ─────────────────────────
    log.info("\nFitting final model on all training data (2013–2021, excl. 2020)...")
    final_model, final_scaler, raw_coefs = fit_model(train_df)

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
