"""
shap_new_data.py — Run SHAP feature selection on the new 2008–2026 dataset.

Uses the same temporal CV logic as shap_selector.py but feeds features_new.csv
(from new_data_loader.py) instead of features_candidates.csv.

With 16 years of data (2008–2024) and ~1000 tournament team-seasons vs 668
before, previously unstable features may now show consistent signal.

CV folds: held-out years 2016–2024 (train on all prior years).
Output:
  data/processed/shap_new_results.csv    — full SHAP table
  data/processed/shap_new_selected.csv   — selected features only
  data/processed/shap_new_importance.png — bar chart

Usage:
    python3 -m src.models.shap_new_data
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import PROCESSED_DIR
from src.features.new_data_loader import load_new_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Thresholds (same as shap_selector.py) ─────────────────────────────────────
IMPORTANCE_THRESHOLD  = 0.01
CONSISTENCY_THRESHOLD = 0.65
MIN_PRIOR_SEASONS     = 3

CV_YEARS = list(range(2016, 2025))  # held-out one at a time

XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     42,
    "verbosity":        0,
}

# Columns that are not features (identifiers / targets)
NON_FEATURE_COLS = {"YEAR", "TEAM", "SEED", "CONF", "ROUNDS_WON",
                    # Evan Miya cols too sparse before 2022 — excluded
                    "KILLSHOTS PER GAME", "KILL SHOTS CONCEDED PER GAME",
                    "KILLSHOTS MARGIN", "ROSTER RANK", "INJURY RANK",
                    "RELATIVE RATING"}


def run_shap_cv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temporal CV SHAP feature selection on team-level data.

    For each hold-out year (2016–2024):
      1. Train XGBoost on all prior years → predict ROUNDS_WON.
      2. Compute SHAP values on held-out year.
      3. Record mean |SHAP| and sign fraction per feature.

    Args:
        df: features_new DataFrame with ROUNDS_WON column.

    Returns:
        DataFrame [feature, mean_abs_shap, sign_consistency, n_folds].
    """
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    all_years = sorted(df["YEAR"].unique())
    log.info(f"Running SHAP CV on {len(feat_cols)} features, years {all_years}")

    fold_abs:  dict[str, list[float]] = {f: [] for f in feat_cols}
    fold_sign: dict[str, list[float]] = {f: [] for f in feat_cols}
    fold_count = 0

    for year in CV_YEARS:
        prior = [y for y in all_years if y < year]
        if len(prior) < MIN_PRIOR_SEASONS:
            log.info(f"  Skipping {year}: only {len(prior)} prior seasons")
            continue

        train = df[df["YEAR"].isin(prior)].dropna(subset=feat_cols + ["ROUNDS_WON"])
        test  = df[(df["YEAR"] == year) & df["ROUNDS_WON"].notna()].copy()
        test  = test.dropna(subset=feat_cols)

        if len(train) < 30 or len(test) == 0:
            log.info(f"  Skipping {year}: train={len(train)}, test={len(test)}")
            continue

        X_train = train[feat_cols].values.astype(float)
        y_train = train["ROUNDS_WON"].values.astype(float)
        X_test  = test[feat_cols].values.astype(float)

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train, verbose=False)

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)

        for i, feat in enumerate(feat_cols):
            col_shap = shap_vals[:, i]
            fold_abs[feat].append(float(np.mean(np.abs(col_shap))))
            fold_sign[feat].append(float(np.mean(col_shap > 0)))

        fold_count += 1
        log.info(f"  Fold {year}: {len(prior)} prior seasons, {len(train)} train, {len(test)} test")

    log.info(f"Temporal CV complete: {fold_count} folds")

    records = []
    for feat in feat_cols:
        abs_vals  = fold_abs[feat]
        sign_vals = fold_sign[feat]
        if not abs_vals:
            continue
        records.append({
            "feature":          feat,
            "mean_abs_shap":    float(np.mean(abs_vals)),
            "sign_consistency": float(np.mean(sign_vals)),
            "n_folds":          len(abs_vals),
        })

    return pd.DataFrame(records).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def select_features(shap_df: pd.DataFrame) -> list[str]:
    """
    Select features that are both important AND directionally stable.

    Criteria:
      1. mean_abs_shap > IMPORTANCE_THRESHOLD
      2. sign_consistency > CONSISTENCY_THRESHOLD or < (1 - CONSISTENCY_THRESHOLD)

    Args:
        shap_df: Output of run_shap_cv().

    Returns:
        Ordered list of selected feature names (most important first).
    """
    selected = shap_df[
        (shap_df["mean_abs_shap"] > IMPORTANCE_THRESHOLD) &
        (
            (shap_df["sign_consistency"] > CONSISTENCY_THRESHOLD) |
            (shap_df["sign_consistency"] < (1 - CONSISTENCY_THRESHOLD))
        )
    ]
    feats = selected["feature"].tolist()
    log.info(f"Selected {len(feats)}/{len(shap_df)} features")
    return feats


def plot_results(shap_df: pd.DataFrame, selected: list[str], out_path: Path) -> None:
    """Save a bar chart of SHAP importance with selected features highlighted."""
    top = shap_df.head(25).copy()
    colors = ["#22d3a0" if f in selected else "#444" for f in top["feature"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#111")

    bars = ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color=colors[::-1])
    ax.set_xlabel("Mean |SHAP|", color="#aaa")
    ax.set_title("SHAP Feature Importance — New Dataset (2008–2024)", color="#f2f2f2", fontsize=13)
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"Saved plot to {out_path}")


def main() -> list[str]:
    df = load_new_features(save=False)

    # Only use years with actual results for CV
    df_train = df[df["ROUNDS_WON"].notna()].copy()
    log.info(f"Training dataset: {len(df_train)} team-seasons with ROUNDS_WON")

    shap_df = run_shap_cv(df_train)

    # Save full results
    shap_df.to_csv(PROCESSED_DIR / "shap_new_results.csv", index=False)
    log.info(f"\n{'='*60}")
    log.info("SHAP RESULTS — NEW DATASET")
    log.info(f"{'='*60}")
    log.info(f"\n{shap_df.head(20).to_string(index=False)}")

    selected = select_features(shap_df)
    selected_df = shap_df[shap_df["feature"].isin(selected)]
    selected_df.to_csv(PROCESSED_DIR / "shap_new_selected.csv", index=False)

    plot_results(shap_df, selected, PROCESSED_DIR / "shap_new_importance.png")

    print(f"\n{'='*60}")
    print(f"SELECTED FEATURES ({len(selected)}):")
    print(f"{'='*60}")
    for f in selected:
        row = shap_df[shap_df["feature"] == f].iloc[0]
        direction = "↑" if row["sign_consistency"] > 0.5 else "↓"
        print(f"  {direction} {f:30s}  shap={row['mean_abs_shap']:.4f}  consistency={row['sign_consistency']:.2f}")

    return selected


if __name__ == "__main__":
    main()
