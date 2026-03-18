"""
shap_feature_selection.py — Data-driven feature selection via SHAP.

Problem we're solving
---------------------
Manually deciding which features to include is guesswork. We dropped QMS,
COACH_PREMIUM, ADJOE, ADJDE because they hurt CV accuracy — but that could
mean they're noisy OR that collinearity was the issue, not the features
themselves. We can't tell which without measuring actual feature contributions.

SHAP (SHapley Additive exPlanations) measures the actual contribution of each
feature to each individual prediction. Unlike permutation importance or
correlation coefficients, SHAP values:
  - Are computed per-prediction, not per-feature globally
  - Handle feature interactions correctly
  - Show DIRECTION (positive = pushed toward win, negative = away from win)
  - Are theoretically grounded (Shapley values from cooperative game theory)

Approach
--------
1. Start with ALL candidate features (everything in features_coaching.csv
   that's numeric and has reasonable coverage)
2. For each temporal CV fold (train on years < test year):
   a. Fit L2 logistic regression on training data
   b. Compute SHAP values on the TEST set (held-out, no leakage)
   c. Record mean |SHAP| per feature for this fold
3. Aggregate across folds:
   - mean_abs_shap:  average magnitude across folds (importance signal)
   - sign_consistency: fraction of folds where mean SHAP > 0 (stability)
     values near 0.0 or 1.0 = consistent direction
     values near 0.5 = feature flips direction across years = unreliable
4. Select features where:
   - mean_abs_shap >= SHAP_IMPORTANCE_THRESHOLD
   - sign_consistency <= 0.2 or >= 0.8  (consistent direction)
5. Retrain and compare CV log loss with full vs selected feature set

Output
------
  - data/processed/shap_feature_importance.csv — full ranking table
  - data/processed/shap_feature_importance.png — bar chart
  - Prints recommended FEATURES list to paste into formula_model.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── All candidate features ─────────────────────────────────────────────────────
# Everything that's numeric, has reasonable coverage, and is not a direct
# proxy for another feature (e.g. SEED is excluded because it's r=0.91 with TQS
# and adds no new information — TQS already captures team strength better).
# KENPOM_NETRTG excluded — same signal as TRUE_QUALITY_SCORE before luck correction.
CANDIDATE_FEATURES = [
    "TRUE_QUALITY_SCORE",   # AdjEM - 0.4*Luck (primary strength signal)
    "SEED_DIVERGENCE",      # actual_seed - implied_seed (upset identifier)
    "QMS",                  # quality momentum score (recent form vs good teams)
    "COACH_PREMIUM",        # career tournament wins above seed expectation
    "ADJOE",                # adjusted offensive efficiency
    "ADJDE",                # adjusted defensive efficiency
    "BARTHAG",              # power rating (independent of KenPom)
    "EFG_O",                # effective field goal % (shooting quality)
    "EFG_D",                # effective field goal % allowed (defense quality)
    "TOR",                  # turnover rate
    "TORD",                 # turnover rate forced
    "ORB",                  # offensive rebound rate
    "DRB",                  # defensive rebound rate
    "ADJ_T",                # adjusted tempo
]

# SHAP selection thresholds
# With only 8 temporal folds and ~60-70 test games per fold, SHAP values are
# inherently noisy. We use relaxed thresholds: a feature that flips direction
# in 2 of 8 folds (0.25/0.75) is still meaningfully consistent.
SHAP_IMPORTANCE_THRESHOLD = 0.01   # mean |SHAP| below this → drop
SIGN_CONSISTENCY_LOW  = 0.30       # <= this → consistently negative (keep)
SIGN_CONSISTENCY_HIGH = 0.70       # >= this → consistently positive (keep)
# Features where sign_consistency is between 0.30 and 0.70 flip direction
# too often across years — the model can't decide which way this feature points.
# Those are dropped as unreliable.

MIN_TRAIN_SEASONS = 3
C_FOR_SELECTION   = 1.0   # fixed C during selection (not grid-searched, for speed)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Load matchup data with the full candidate feature set.

    Returns symmetric matchup DataFrame: for each game (A beats B),
    two rows with feature differences (A-B, label=1) and (B-A, label=0).
    Only includes years/games where ALL candidate features are present.
    """
    from src.models.formula_model import load_matchup_data, FEATURES as DIFF_FEATURES

    # Load using formula_model pipeline (features_candidates.csv preferred)
    # Then restrict to CANDIDATE_FEATURES that are actually in the data
    base_df = load_matchup_data()

    # Check which candidates are available
    available = [f for f in CANDIDATE_FEATURES if f in base_df.columns]
    missing = [f for f in CANDIDATE_FEATURES if f not in base_df.columns]
    if missing:
        log.warning(f"Candidate features not in matchup data: {missing}")

    log.info(f"Available candidate features: {available}")
    return base_df, available


# ── SHAP computation ───────────────────────────────────────────────────────────

def compute_shap_per_fold(
    df: pd.DataFrame,
    features: list[str],
    min_train_seasons: int = MIN_TRAIN_SEASONS,
    c: float = C_FOR_SELECTION,
) -> pd.DataFrame:
    """
    Compute SHAP values for each temporal CV fold and aggregate.

    For each test year (with at least min_train_seasons prior years):
      1. Fit L2 LR on training years
      2. Compute LinearExplainer SHAP values on test set
      3. Record mean |SHAP| and mean SHAP (for sign) per feature

    Uses LinearExplainer (exact for linear models) rather than KernelExplainer
    (which would be prohibitively slow for 668 samples × 14 features).

    Args:
        df:                 Symmetric matchup DataFrame.
        features:           List of feature names to evaluate.
        min_train_seasons:  Minimum prior seasons required.
        c:                  L2 regularization strength.

    Returns:
        DataFrame with one row per (year, feature) with columns:
        year, feature, mean_abs_shap, mean_shap, n_test_samples.
    """
    import shap

    seasons = sorted(df["YEAR"].unique())
    records = []

    for year in seasons:
        prior = [y for y in seasons if y < year and y != 2020]
        if len(prior) < min_train_seasons:
            continue

        train = df[df["YEAR"].isin(prior)].dropna(subset=features + ["LABEL"])
        test  = df[df["YEAR"] == year].dropna(subset=features + ["LABEL"])

        if len(train) < 50 or len(test) == 0:
            continue

        X_train = train[features].values.astype(float)
        y_train = train["LABEL"].values
        X_test  = test[features].values.astype(float)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        model = LogisticRegression(
            solver="lbfgs", C=c, max_iter=2000, random_state=42
        )
        model.fit(X_train_s, y_train)

        # LinearExplainer uses the closed-form solution for linear models:
        # SHAP_i = coef_i * (x_i - E[x_i])
        # This is exact (no sampling approximation) and fast.
        explainer = shap.LinearExplainer(model, X_train_s, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_test_s)
        # shap_values shape: (n_test, n_features)
        # Values are in scaled (standardized) space — we un-scale for interpretability
        # by multiplying by 1/scaler.scale_ per feature.
        shap_raw = shap_values / scaler.scale_

        for fi, feat in enumerate(features):
            col_shap = shap_raw[:, fi]
            records.append({
                "year":           year,
                "feature":        feat,
                "mean_abs_shap":  float(np.abs(col_shap).mean()),
                "mean_shap":      float(col_shap.mean()),
                "n_test_samples": len(col_shap),
            })

        log.info(
            f"  SHAP fold {year}: {len(test)//2} test games, "
            f"{len(prior)} train seasons"
        )

    return pd.DataFrame(records)


def aggregate_shap(fold_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-fold SHAP values into a feature-level summary.

    Args:
        fold_df: Output of compute_shap_per_fold().

    Returns:
        DataFrame indexed by feature with columns:
          mean_abs_shap     — average |SHAP| across folds (importance)
          std_abs_shap      — std dev of |SHAP| (stability)
          mean_shap         — average signed SHAP (direction)
          sign_consistency  — fraction of folds where mean_shap > 0
                              0.0=always negative, 1.0=always positive, 0.5=flips
          n_folds           — number of CV folds this feature appeared in
          recommended       — True if passes both importance and consistency filters
    """
    agg = fold_df.groupby("feature").agg(
        mean_abs_shap=("mean_abs_shap", "mean"),
        std_abs_shap=("mean_abs_shap", "std"),
        mean_shap=("mean_shap", "mean"),
        n_folds=("year", "count"),
    ).reset_index()

    # Sign consistency: fraction of folds where feature pushed toward win
    sign_by_fold = (
        fold_df.groupby(["feature", "year"])["mean_shap"]
        .mean()
        .reset_index()
    )
    sign_consistency = (
        sign_by_fold.groupby("feature")["mean_shap"]
        .apply(lambda x: (x > 0).mean())
        .rename("sign_consistency")
        .reset_index()
    )
    agg = agg.merge(sign_consistency, on="feature")

    # Apply selection criteria
    agg["passes_importance"] = agg["mean_abs_shap"] >= SHAP_IMPORTANCE_THRESHOLD
    agg["passes_consistency"] = (
        (agg["sign_consistency"] <= SIGN_CONSISTENCY_LOW) |
        (agg["sign_consistency"] >= SIGN_CONSISTENCY_HIGH)
    )
    agg["recommended"] = agg["passes_importance"] & agg["passes_consistency"]

    return agg.sort_values("mean_abs_shap", ascending=False)


# ── Validation: compare feature sets ──────────────────────────────────────────

def compare_feature_sets(
    df: pd.DataFrame,
    full_features: list[str],
    selected_features: list[str],
    baseline_features: list[str],
    c: float = C_FOR_SELECTION,
    min_train_seasons: int = MIN_TRAIN_SEASONS,
) -> pd.DataFrame:
    """
    Compare temporal CV log loss and accuracy across three feature sets:
      - baseline (current 2-feature model)
      - full (all candidates)
      - selected (SHAP-filtered subset)

    Args:
        df:                  Symmetric matchup DataFrame.
        full_features:       All candidate features.
        selected_features:   SHAP-selected subset.
        baseline_features:   Current production features.
        c:                   L2 C for all comparisons.
        min_train_seasons:   Minimum prior seasons.

    Returns:
        DataFrame with per-fold metrics for all three feature sets.
    """
    seasons = sorted(df["YEAR"].unique())
    records = []

    for year in seasons:
        prior = [y for y in seasons if y < year and y != 2020]
        if len(prior) < min_train_seasons:
            continue

        for label, feats in [
            ("baseline_2feat", baseline_features),
            ("full_all_feats", full_features),
            ("shap_selected",  selected_features),
        ]:
            train = df[df["YEAR"].isin(prior)].dropna(subset=feats + ["LABEL"])
            test  = df[df["YEAR"] == year].dropna(subset=feats + ["LABEL"])
            if len(train) < 50 or len(test) == 0:
                continue

            X_train = train[feats].values.astype(float)
            y_train = train["LABEL"].values
            X_test  = test[feats].values.astype(float)
            y_test  = test["LABEL"].values

            scaler = StandardScaler()
            model  = LogisticRegression(solver="lbfgs", C=c, max_iter=2000, random_state=42)
            model.fit(scaler.fit_transform(X_train), y_train)
            probs = model.predict_proba(scaler.transform(X_test))[:, 1]
            preds = (probs >= 0.5).astype(int)

            records.append({
                "year":        year,
                "feature_set": label,
                "n_features":  len(feats),
                "accuracy":    round(accuracy_score(y_test, preds), 4),
                "log_loss":    round(log_loss(y_test, np.clip(probs, 1e-7, 1-1e-7)), 4),
            })

    return pd.DataFrame(records)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_shap_importance(
    agg: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Plot mean |SHAP| per feature with color coding for recommended/dropped.

    Args:
        agg:      Output of aggregate_shap().
        out_path: Path to save PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: mean |SHAP| bar chart
    colors = ["#2ecc71" if r else "#e74c3c" for r in agg["recommended"]]
    axes[0].barh(agg["feature"], agg["mean_abs_shap"], color=colors, alpha=0.85)
    axes[0].axvline(
        SHAP_IMPORTANCE_THRESHOLD, color="black", linestyle="--", linewidth=1,
        label=f"Threshold ({SHAP_IMPORTANCE_THRESHOLD})"
    )
    axes[0].set_xlabel("Mean |SHAP| across CV folds")
    axes[0].set_title("Feature Importance (SHAP)\nGreen = selected, Red = dropped")
    axes[0].legend(fontsize=9)
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis="x")

    # Right: sign consistency
    consistency_colors = []
    for sc in agg["sign_consistency"]:
        if sc <= SIGN_CONSISTENCY_LOW or sc >= SIGN_CONSISTENCY_HIGH:
            consistency_colors.append("#2ecc71")  # consistent
        else:
            consistency_colors.append("#e74c3c")   # flips direction

    axes[1].barh(agg["feature"], agg["sign_consistency"], color=consistency_colors, alpha=0.85)
    axes[1].axvline(SIGN_CONSISTENCY_LOW,  color="black", linestyle="--", linewidth=1)
    axes[1].axvline(SIGN_CONSISTENCY_HIGH, color="black", linestyle="--", linewidth=1,
                    label=f"Consistency bounds ({SIGN_CONSISTENCY_LOW}/{SIGN_CONSISTENCY_HIGH})")
    axes[1].axvline(0.5, color="gray", linestyle=":", linewidth=1, label="50% (random flip)")
    axes[1].set_xlabel("Sign consistency (fraction of folds where SHAP > 0)")
    axes[1].set_title("Direction Stability\nGreen = consistent, Red = flips across years")
    axes[1].set_xlim(0, 1)
    axes[1].legend(fontsize=9)
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info(f"SHAP importance plot saved to {out_path}")
    plt.close(fig)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> list[str]:
    """
    Run SHAP feature selection and return the recommended feature list.

    Steps:
      1. Load all candidate features
      2. Compute SHAP values per temporal CV fold
      3. Aggregate and filter
      4. Compare log loss across baseline / full / selected feature sets
      5. Save results and plot
      6. Print recommended FEATURES list to paste into formula_model.py

    Returns:
        List of recommended feature names (in importance order).
    """
    log.info("Loading matchup data with full candidate feature set...")
    base_df, available_features = load_data()

    # Filter to years with data and exclude 2020
    df = base_df[
        base_df["YEAR"].between(2013, 2024) & (base_df["YEAR"] != 2020)
    ].copy()
    log.info(f"Dataset: {len(df)//2} games, {df['YEAR'].nunique()} seasons")

    # ── Step 1: SHAP per fold ─────────────────────────────────────────────────
    log.info(f"\nComputing SHAP values across temporal CV folds...")
    log.info(f"  Candidate features ({len(available_features)}): {available_features}")
    fold_df = compute_shap_per_fold(df, available_features)

    # ── Step 2: Aggregate ─────────────────────────────────────────────────────
    agg = aggregate_shap(fold_df)

    print("\n" + "=" * 85)
    print("SHAP FEATURE IMPORTANCE (temporal CV, mean across held-out folds)")
    print("=" * 85)
    print(f"  {'Feature':<22} {'Mean|SHAP|':>10}  {'Std|SHAP|':>10}  "
          f"{'SignConsist':>12}  {'Folds':>6}  {'Keep?':>6}")
    print("  " + "-" * 75)
    for _, row in agg.iterrows():
        keep = "✓ YES" if row["recommended"] else "✗ NO "
        flag = ""
        if not row["passes_importance"]:
            flag = "(low importance)"
        elif not row["passes_consistency"]:
            flag = "(sign flips)"
        print(f"  {row['feature']:<22} {row['mean_abs_shap']:>10.4f}  "
              f"{row['std_abs_shap']:>10.4f}  {row['sign_consistency']:>12.3f}  "
              f"{int(row['n_folds']):>6}  {keep}  {flag}")

    selected = agg[agg["recommended"]]["feature"].tolist()

    print(f"\n  Selected {len(selected)} of {len(available_features)} features:")
    for f in selected:
        print(f"    - {f}")

    # ── Step 3: Compare feature sets ─────────────────────────────────────────
    baseline = ["TRUE_QUALITY_SCORE", "SEED_DIVERGENCE"]

    log.info("\nComparing CV performance across feature sets...")
    # Fall back to top-2 SHAP features if selection picked nothing
    if not selected:
        log.warning("SHAP selected 0 features — using top-2 by mean|SHAP| as fallback")
        selected = agg.head(2)["feature"].tolist()
    comparison = compare_feature_sets(
        df, available_features, selected, baseline
    )

    print("\n" + "=" * 70)
    print("FEATURE SET COMPARISON (temporal CV log loss + accuracy)")
    print("=" * 70)
    summary = (
        comparison.groupby("feature_set")
        .agg(mean_ll=("log_loss", "mean"), mean_acc=("accuracy", "mean"),
             n_folds=("year", "count"))
        .reset_index()
        .sort_values("mean_ll")
    )
    print(f"  {'Feature Set':<20} {'Features':>9}  {'Mean LL':>9}  {'Mean Acc':>9}  {'Folds':>6}")
    print("  " + "-" * 60)

    feat_counts = {"baseline_2feat": len(baseline),
                   "full_all_feats": len(available_features),
                   "shap_selected":  len(selected)}
    for _, row in summary.iterrows():
        marker = " ← best" if row["feature_set"] == summary.iloc[0]["feature_set"] else ""
        print(f"  {row['feature_set']:<20} {feat_counts[row['feature_set']]:>9}  "
              f"{row['mean_ll']:>9.4f}  {row['mean_acc']:>9.4f}  "
              f"{int(row['n_folds']):>6}{marker}")

    # Per-fold detail
    pivot = comparison.pivot_table(
        index="year", columns="feature_set", values="log_loss"
    ).reset_index()
    print(f"\n  Per-fold log loss:")
    print(f"  {'Year':<6} {'baseline':>10}  {'full':>10}  {'shap_sel':>10}  {'best':>10}")
    print("  " + "-" * 50)
    for _, row in pivot.iterrows():
        vals = {
            "baseline": row.get("baseline_2feat", float("nan")),
            "full":     row.get("full_all_feats", float("nan")),
            "shap_sel": row.get("shap_selected",  float("nan")),
        }
        best = min(vals, key=lambda k: vals[k] if not np.isnan(vals[k]) else float("inf"))
        print(f"  {int(row['year']):<6} "
              f"{vals['baseline']:>10.4f}  "
              f"{vals['full']:>10.4f}  "
              f"{vals['shap_sel']:>10.4f}  "
              f"{best:>10}")

    # ── Step 4: Save outputs ──────────────────────────────────────────────────
    out_csv = PROCESSED_DIR / "shap_feature_importance.csv"
    agg.to_csv(out_csv, index=False)
    log.info(f"SHAP importance table saved to {out_csv}")

    fold_df.to_csv(PROCESSED_DIR / "shap_fold_details.csv", index=False)
    comparison.to_csv(PROCESSED_DIR / "shap_feature_comparison.csv", index=False)

    plot_shap_importance(agg, PROCESSED_DIR / "shap_feature_importance.png")

    # ── Step 5: Print copy-paste output ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("PASTE THIS INTO formula_model.py → FEATURES list:")
    print("=" * 65)
    print("FEATURES = [")
    for f in selected:
        row = agg[agg["feature"] == f].iloc[0]
        direction = "positive" if row["sign_consistency"] >= SIGN_CONSISTENCY_HIGH else "negative"
        print(f'    "{f}",  # mean|SHAP|={row["mean_abs_shap"]:.4f}, '
              f'sign={direction} ({row["sign_consistency"]:.2f} consistency)')
    print("]")

    print(f"\n  Saved:")
    print(f"    {out_csv}")
    print(f"    {PROCESSED_DIR / 'shap_feature_importance.png'}")
    print(f"    {PROCESSED_DIR / 'shap_feature_comparison.csv'}")

    return selected


if __name__ == "__main__":
    selected = main()
