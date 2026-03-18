"""
exp_EXPERIMENT_NAME.py — [One-line description of what this experiment tests]

Hypothesis:
  [What do you expect to happen and why?]
  E.g.: "Adding BARTHAG alongside TQS should reduce log loss because Torvik
  and KenPom capture slightly different variance in team strength."

Success criterion:
  - Mean CV log loss < 0.451 (current baseline)
  - Holdout mean ESPN > 830 (current baseline)
  - Champion accuracy >= 1/3

Baseline (run: python3 -m src.models.formula_model, 2026-03-17):
  Features: TRUE_QUALITY_SCORE, BARTHAG, ADJ_T, DRB, EFF_RATIO, 3P_O, FTR
  Mean CV acc:  0.780,  mean CV log loss: 0.451
  ESPN: 2022=640, 2023=660, 2024=1190  → mean=830
  Champion: 1/3
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import PROCESSED_DIR

# ── Experiment configuration ──────────────────────────────────────────────────

# Change this to test your hypothesis:
EXPERIMENT_FEATURES = [
    "TRUE_QUALITY_SCORE",
    "BARTHAG",
    "ADJ_T",
    "DRB",
    "EFF_RATIO",
    "3P_O",
    "FTR",
    # ADD / REMOVE features here
]

# Baseline (do not change — used for comparison)
BASELINE_FEATURES = [
    "TRUE_QUALITY_SCORE",
    "BARTHAG",
    "ADJ_T",
    "DRB",
    "EFF_RATIO",
    "3P_O",
    "FTR",
]

HOLDOUT_YEARS = [2022, 2023, 2024]
TRAIN_YEARS   = list(range(2013, 2022))   # 2013–2021, excl. 2020
C_GRID        = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
MIN_PRIOR     = 3


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(features: list[str]) -> pd.DataFrame:
    """Load matchup data with the specified feature set."""
    from src.models.formula_model import load_matchup_data as _base_load
    from src.models.win_probability import _build_kaggle_to_cbb_map
    import pandas as pd

    candidates_path = PROCESSED_DIR / "features_candidates.csv"
    coaching_path   = PROCESSED_DIR / "features_coaching.csv"
    feats_df = pd.read_csv(candidates_path if candidates_path.exists() else coaching_path)

    results = pd.read_csv(ROOT / "data/external/kaggle/MNCAATourneyDetailedResults.csv")
    teams   = pd.read_csv(ROOT / "data/external/kaggle/MTeams.csv")
    kaggle_to_cbb = _build_kaggle_to_cbb_map(feats_df, teams)

    feat_lookup = {}
    for _, row in feats_df.iterrows():
        if all(f in row.index for f in features):
            feat_lookup[(row["TEAM"], int(row["YEAR"]))] = row[features].values.astype(float)

    records = []
    for _, game in results.iterrows():
        year  = int(game["Season"])
        if not (2013 <= year <= 2024) or year == 2020:
            continue
        w = kaggle_to_cbb.get(int(game["WTeamID"]))
        l = kaggle_to_cbb.get(int(game["LTeamID"]))
        if not w or not l:
            continue
        wf = feat_lookup.get((w, year))
        lf = feat_lookup.get((l, year))
        if wf is None or lf is None:
            continue
        diff = wf - lf
        if np.any(np.isnan(diff)):
            continue
        records.append({**dict(zip(features, diff)),  "LABEL": 1, "YEAR": year})
        records.append({**dict(zip(features, -diff)), "LABEL": 0, "YEAR": year})

    return pd.DataFrame(records)


# ── CV harness ────────────────────────────────────────────────────────────────

def run_cv(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, float]:
    """Temporal CV with C grid search. Returns per-fold results and best C."""
    all_years = sorted(y for y in df["YEAR"].unique() if y != 2020)
    best_c, best_mean_ll = C_GRID[0], float("inf")
    all_records: dict[float, list[dict]] = {}

    for c in C_GRID:
        records = []
        for year in all_years:
            prior = [y for y in all_years if y < year]
            if len(prior) < MIN_PRIOR:
                continue
            train = df[df["YEAR"].isin(prior)].dropna(subset=features + ["LABEL"])
            test  = df[df["YEAR"] == year].dropna(subset=features + ["LABEL"])
            if len(train) < 50 or len(test) == 0:
                continue

            scaler = StandardScaler()
            X_train = scaler.fit_transform(train[features].values)
            X_test  = scaler.transform(test[features].values)
            y_train = train["LABEL"].values
            y_test  = test["LABEL"].values

            model = LogisticRegression(solver="lbfgs", C=c, max_iter=2000, random_state=42)
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]

            records.append({
                "year":     year,
                "accuracy": accuracy_score(y_test, (probs >= 0.5).astype(int)),
                "log_loss": log_loss(y_test, np.clip(probs, 1e-7, 1 - 1e-7)),
                "C":        c,
            })
        if not records:
            continue
        mean_ll = float(np.mean([r["log_loss"] for r in records]))
        all_records[c] = records
        if mean_ll < best_mean_ll:
            best_mean_ll = mean_ll
            best_c = c

    return pd.DataFrame(all_records.get(best_c, [])), best_c


# ── Holdout ESPN scoring ───────────────────────────────────────────────────────

def score_holdout(features: list[str], best_c: float) -> list[dict]:
    """Train on 2013-2021, predict holdout years, compute ESPN scores."""
    from src.models.formula_model import fit_model, simulate_bracket

    df = load_data(features)
    train_df = df[df["YEAR"].isin([y for y in TRAIN_YEARS if y != 2020])]

    # Temporarily override FEATURES in formula_model for simulate_bracket
    import src.models.formula_model as fm
    _orig = fm.FEATURES
    fm.FEATURES = features

    model, scaler, _ = fit_model(train_df, c=best_c)
    results = []
    for year in HOLDOUT_YEARS:
        r = simulate_bracket(year, model, scaler)
        results.append({
            "year": year,
            "espn_score": r["espn_score"],
            "champion": r["champion"],
        })

    fm.FEATURES = _orig  # restore
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print(f"EXPERIMENT: {__file__.split('/')[-1]}")
    print("=" * 65)

    # Run CV for experiment features
    print(f"\nExperiment features ({len(EXPERIMENT_FEATURES)}): {EXPERIMENT_FEATURES}")
    print("Running temporal CV...")
    df_exp = load_data(EXPERIMENT_FEATURES)
    cv_exp, c_exp = run_cv(df_exp, EXPERIMENT_FEATURES)

    # Run CV for baseline features
    print(f"\nBaseline features ({len(BASELINE_FEATURES)}): {BASELINE_FEATURES}")
    print("Running temporal CV...")
    df_base = load_data(BASELINE_FEATURES)
    cv_base, c_base = run_cv(df_base, BASELINE_FEATURES)

    # Compare CV
    print("\n" + "=" * 65)
    print("CV COMPARISON")
    print("=" * 65)
    print(f"{'Metric':<20} {'Baseline':>12} {'Experiment':>12} {'Delta':>10}")
    print("-" * 58)
    for metric in ["accuracy", "log_loss"]:
        b = cv_base[metric].mean() if len(cv_base) else float("nan")
        e = cv_exp[metric].mean()  if len(cv_exp)  else float("nan")
        delta = e - b
        flag = " ✓" if (metric == "accuracy" and delta > 0) or (metric == "log_loss" and delta < 0) else " ✗"
        print(f"{metric:<20} {b:>12.4f} {e:>12.4f} {delta:>+10.4f}{flag}")

    # Score holdout
    print("\nScoring holdout years (2022-2024)...")
    espn_exp  = score_holdout(EXPERIMENT_FEATURES, c_exp)
    espn_base = score_holdout(BASELINE_FEATURES,   c_base)

    print("\n" + "=" * 65)
    print("HOLDOUT ESPN COMPARISON")
    print("=" * 65)
    print(f"{'Year':<8} {'Baseline':>10} {'Experiment':>12} {'Delta':>10} {'Champ (exp)':<20}")
    print("-" * 65)
    for b, e in zip(espn_base, espn_exp):
        delta = (e["espn_score"] or 0) - (b["espn_score"] or 0)
        flag = " ✓" if delta > 0 else " ✗"
        print(f"{b['year']:<8} {b['espn_score'] or 'N/A':>10} {e['espn_score'] or 'N/A':>12} "
              f"{delta:>+10}{flag}  {e['champion']}")

    b_mean = np.mean([r["espn_score"] for r in espn_base if r["espn_score"]])
    e_mean = np.mean([r["espn_score"] for r in espn_exp  if r["espn_score"]])
    print(f"\n{'MEAN':<8} {b_mean:>10.0f} {e_mean:>12.0f} {e_mean-b_mean:>+10.0f}")

    print("\n" + "=" * 65)
    print("VERDICT")
    print("=" * 65)
    cv_better  = cv_exp["log_loss"].mean() < cv_base["log_loss"].mean() if len(cv_exp) and len(cv_base) else False
    espn_better = e_mean > b_mean
    if cv_better and espn_better:
        print("✓ ADOPT — both CV and holdout ESPN improved")
        print(f"  Paste into formula_model.py FEATURES:")
        print(f"  FEATURES = {EXPERIMENT_FEATURES!r}")
    elif espn_better:
        print("⚠ CONSIDER — ESPN improved but CV log_loss did not")
        print("  Investigate which years drove the improvement")
    elif cv_better:
        print("✗ REJECT — CV improved but ESPN dropped (likely hurts champion picks)")
    else:
        print("✗ REJECT — both metrics worse")


if __name__ == "__main__":
    main()
