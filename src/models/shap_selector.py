"""
shap_selector.py — XGBoost + SHAP team-level feature selection.

WHY TEAM-LEVEL (not matchup-level)?

Previous approach: feature(A) - feature(B) → predict game outcome (0/1)
Problem: a single game is a coin flip. 623 games with huge variance =
  SHAP sign flips every year. No feature shows consistent direction.

This approach: raw team features → predict rounds_won (0-6)
Why better:
  - Target integrates over the whole tournament run (6 games, not 1)
  - Rounds won is far less noisy than a single binary outcome
  - SHAP on raw features is interpretable: "TQS pushes up rounds won"
  - Still enforces temporal CV: train on prior years only

After selecting stable features here, formula_model.py still uses
matchup differences — but now we know WHICH features are informative
before differencing them.

Selection criteria (both must pass):
  1. mean |SHAP| > IMPORTANCE_THRESHOLD (0.01)
  2. Sign consistency > CONSISTENCY_THRESHOLD (0.65)
     OR sign consistency < (1 - CONSISTENCY_THRESHOLD)
     — feature consistently pushes rounds_won up or down across years

Output:
  data/processed/shap_candidates.png       — importance + direction chart
  data/processed/shap_selected_features.csv — ranked selected features
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

from config import PROCESSED_DIR, EXTERNAL_DIR
from src.features.candidates import build_candidate_features, CANDIDATE_FEATURES
from src.utils.team_names import build_daynum_to_round as _build_daynum_to_round

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

IMPORTANCE_THRESHOLD  = 0.01
CONSISTENCY_THRESHOLD = 0.65

XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        3,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "objective":        "reg:squarederror",  # regression — rounds won is ordinal
    "random_state":     42,
}

MIN_PRIOR_SEASONS = 3
CV_YEARS = list(range(2016, 2025))


# ── Build team-level target: rounds won ───────────────────────────────────────

def build_rounds_won(year: int, results_df: pd.DataFrame,
                     kaggle_to_cbb: dict[int, str]) -> dict[str, int]:
    """
    Compute rounds_won for every tournament team in a given year.

    rounds_won = number of games won = furthest round reached.
    Champion = 6, runner-up = 5, F4 losers = 4, etc.
    First-round losers = 0.

    Args:
        year:         Tournament season.
        results_df:   Full MNCAATourneyDetailedResults DataFrame.
        kaggle_to_cbb: Kaggle TeamID → CBB team name map.

    Returns:
        Dict {cbb_team_name: rounds_won} for all teams with result data.
    """
    yr = results_df[results_df["Season"] == year]
    if yr.empty:
        return {}

    daynum_to_round = _build_daynum_to_round(yr)
    rounds_won: dict[str, int] = {}

    for _, row in yr.iterrows():
        rnd = daynum_to_round.get(int(row["DayNum"]))
        if rnd is None:
            continue
        w = kaggle_to_cbb.get(int(row["WTeamID"]))
        l = kaggle_to_cbb.get(int(row["LTeamID"]))
        if w:
            rounds_won[w] = rounds_won.get(w, 0) + 1
        if l and l not in rounds_won:
            rounds_won[l] = 0

    return rounds_won


def build_team_level_dataset(
    features_path: Path = PROCESSED_DIR / "features_candidates.csv",
) -> pd.DataFrame:
    """
    Build a team-season dataset: one row per tournament team per year,
    features = raw candidate stats, target = rounds_won (0-6).

    Skips 2020 (no tournament) and 2025 (no results yet).

    Args:
        features_path: Path to features_candidates.csv. If missing,
                       builds it on-the-fly via build_candidate_features().

    Returns:
        DataFrame with CANDIDATE_FEATURES columns + ROUNDS_WON + YEAR + TEAM.
    """
    from src.utils.team_names import build_kaggle_to_cbb_map as _build_kaggle_to_cbb_map

    # Load or build candidate features
    if features_path.exists():
        feats_df = pd.read_csv(features_path)
        log.info(f"Loaded candidate features from {features_path}")
    else:
        log.info("features_candidates.csv not found — building now...")
        feats_df = build_candidate_features()

    results_path = EXTERNAL_DIR / "kaggle" / "MNCAATourneyDetailedResults.csv"
    teams_path   = EXTERNAL_DIR / "kaggle" / "MTeams.csv"
    results_df   = pd.read_csv(results_path)
    teams_df     = pd.read_csv(teams_path)
    kaggle_to_cbb = _build_kaggle_to_cbb_map(feats_df, teams_df)

    feat_cols = [c for c in CANDIDATE_FEATURES if c in feats_df.columns]

    # Tournament teams only (SEED not null, exclude 2020 + 2025)
    tourney = feats_df[
        feats_df["SEED"].notna() &
        ~feats_df["YEAR"].isin([2020, 2025])
    ].copy()

    records = []
    for year in sorted(tourney["YEAR"].unique()):
        rounds_won = build_rounds_won(year, results_df, kaggle_to_cbb)
        if not rounds_won:
            log.warning(f"  No results found for {year}, skipping")
            continue

        yr_teams = tourney[tourney["YEAR"] == year]
        for _, row in yr_teams.iterrows():
            team = row["TEAM"]
            if team not in rounds_won:
                continue  # team not in results (name mismatch) — skip cleanly
            rec = {col: row[col] for col in feat_cols if col in row.index}
            rec["ROUNDS_WON"] = rounds_won[team]
            rec["YEAR"]       = year
            rec["TEAM"]       = team
            records.append(rec)

    df = pd.DataFrame(records)
    log.info(
        f"Team-level dataset: {len(df)} team-seasons across "
        f"{df['YEAR'].nunique()} years "
        f"(rounds_won range: {df['ROUNDS_WON'].min()}-{df['ROUNDS_WON'].max()})"
    )
    return df


# ── XGBoost SHAP temporal CV ──────────────────────────────────────────────────

def run_shap_temporal_cv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temporal CV: train XGBoost to predict rounds_won on prior years,
    compute SHAP values on held-out year, aggregate across folds.

    For each fold in CV_YEARS:
      1. Train on all team-seasons from prior years.
      2. TreeExplainer SHAP on held-out year team-seasons.
      3. Record mean |SHAP| and sign fraction per feature.

    Aggregates:
      mean_abs_shap    — average importance across folds
      sign_consistency — fraction of folds where mean SHAP > 0
                         (near 0.5 = flipping = unreliable)

    Args:
        df: Output from build_team_level_dataset().

    Returns:
        DataFrame [feature, mean_abs_shap, sign_consistency, n_folds]
        sorted by mean_abs_shap descending.
    """
    feat_cols = [c for c in CANDIDATE_FEATURES if c in df.columns]
    all_years = sorted(df["YEAR"].unique())

    fold_abs:  dict[str, list[float]] = {f: [] for f in feat_cols}
    fold_sign: dict[str, list[float]] = {f: [] for f in feat_cols}

    fold_count = 0
    for year in CV_YEARS:
        prior = [y for y in all_years if y < year]
        if len(prior) < MIN_PRIOR_SEASONS:
            log.info(f"  Skipping {year}: only {len(prior)} prior seasons")
            continue

        train = df[df["YEAR"].isin(prior)].dropna(subset=feat_cols + ["ROUNDS_WON"])
        test  = df[df["YEAR"] == year].dropna(subset=feat_cols + ["ROUNDS_WON"])

        if len(train) < 30 or len(test) == 0:
            log.info(f"  Skipping {year}: insufficient data (train={len(train)}, test={len(test)})")
            continue

        X_train = train[feat_cols].values.astype(float)
        y_train = train["ROUNDS_WON"].values.astype(float)
        X_test  = test[feat_cols].values.astype(float)

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train, verbose=False)

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)
        # shape: (n_teams_in_test_year, n_features)
        # Positive SHAP = pushes rounds_won higher = better predictor of deep runs

        for i, feat in enumerate(feat_cols):
            col_shap = shap_vals[:, i]
            fold_abs[feat].append(float(np.mean(np.abs(col_shap))))
            fold_sign[feat].append(float(np.mean(col_shap > 0)))

        n_test_games = len(test)
        fold_count += 1
        log.info(
            f"  Fold {year}: {len(prior)} prior seasons "
            f"({len(train)} train teams, {n_test_games} test teams)"
        )

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

    return pd.DataFrame(records).sort_values("mean_abs_shap", ascending=False)


# ── Selection ─────────────────────────────────────────────────────────────────

def select_features(shap_df: pd.DataFrame) -> list[str]:
    """
    Select features that are both important AND directionally stable.

    Args:
        shap_df: Output of run_shap_temporal_cv().

    Returns:
        Ordered list of selected feature names.
    """
    mask = (
        (shap_df["mean_abs_shap"] > IMPORTANCE_THRESHOLD) &
        (
            (shap_df["sign_consistency"] > CONSISTENCY_THRESHOLD) |
            (shap_df["sign_consistency"] < (1 - CONSISTENCY_THRESHOLD))
        )
    )
    selected = shap_df[mask]["feature"].tolist()
    log.info(
        f"Selected {len(selected)}/{len(shap_df)} features "
        f"(|SHAP| > {IMPORTANCE_THRESHOLD}, "
        f"consistency > {CONSISTENCY_THRESHOLD} or < {1-CONSISTENCY_THRESHOLD:.2f})"
    )
    return selected


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_shap_results(
    shap_df: pd.DataFrame,
    selected: list[str],
    out_path: Path = PROCESSED_DIR / "shap_candidates.png",
) -> None:
    """
    Two-panel chart: feature importance + sign consistency.

    Left:  mean |SHAP| — green = selected, red = dropped.
    Right: sign consistency — green = stable direction, red = flipping.

    Args:
        shap_df:  Output of run_shap_temporal_cv().
        selected: List of selected feature names.
        out_path: Save path for PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(shap_df) * 0.45)))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    features  = shap_df["feature"].tolist()
    abs_shap  = shap_df["mean_abs_shap"].tolist()
    sign_cons = shap_df["sign_consistency"].tolist()
    colors    = ["#2ea043" if f in selected else "#f85149" for f in features]

    ax = axes[0]
    ax.barh(features, abs_shap, color=colors)
    ax.axvline(IMPORTANCE_THRESHOLD, color="white", linestyle="--", linewidth=1,
               label=f"Threshold ({IMPORTANCE_THRESHOLD})")
    ax.set_title("Feature Importance (SHAP)\nTarget = rounds won per team", pad=10)
    ax.set_xlabel("Mean |SHAP| across CV folds")
    ax.legend(facecolor="#161b22", labelcolor="white", edgecolor="#30363d")
    ax.invert_yaxis()

    ax = axes[1]
    ax.barh(features, sign_cons, color=colors)
    ax.axvline(CONSISTENCY_THRESHOLD, color="white", linestyle="--", linewidth=1,
               label=f"Bounds ({1-CONSISTENCY_THRESHOLD:.2f} / {CONSISTENCY_THRESHOLD:.2f})")
    ax.axvline(1 - CONSISTENCY_THRESHOLD, color="white", linestyle="--", linewidth=1)
    ax.axvline(0.5, color="#8b949e", linestyle=":", linewidth=1, label="50% (random)")
    ax.set_title("Direction Stability\nGreen = consistent, Red = flips across years", pad=10)
    ax.set_xlabel("Fraction of folds where SHAP > 0")
    ax.set_xlim(0, 1)
    ax.legend(facecolor="#161b22", labelcolor="white", edgecolor="#30363d")
    ax.invert_yaxis()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"Chart saved to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> list[str]:
    """
    Full pipeline:
      1. Build team-level dataset (raw features → rounds_won target)
      2. XGBoost + SHAP temporal CV
      3. Select stable features
      4. Save chart + CSV, print FEATURES list for formula_model.py

    Returns:
        List of selected feature names.
    """
    log.info("=" * 65)
    log.info("SHAP Feature Selection — Team-Level (target: rounds won)")
    log.info("=" * 65)

    log.info("\nStep 1: Building team-level dataset...")
    df = build_team_level_dataset()
    log.info(f"  {len(df)} team-seasons, {df['YEAR'].nunique()} years")

    rounds_dist = df.groupby("ROUNDS_WON").size()
    log.info(f"  Rounds won distribution:\n{rounds_dist.to_string()}")

    log.info("\nStep 2: Running XGBoost + SHAP temporal CV...")
    shap_df = run_shap_temporal_cv(df)

    log.info("\nStep 3: Applying selection thresholds...")
    selected = select_features(shap_df)

    print("\n" + "=" * 75)
    print("SHAP FEATURE SELECTION RESULTS  (target = rounds won per team)")
    print("=" * 75)
    print(f"{'Feature':<25} {'mean|SHAP|':>12} {'sign_cons':>10}  {'Status':>10}")
    print("-" * 65)
    for _, row in shap_df.iterrows():
        status = "SELECTED" if row["feature"] in selected else "dropped"
        marker = "✓" if status == "SELECTED" else " "
        print(
            f"{marker} {row['feature']:<23} {row['mean_abs_shap']:>12.5f} "
            f"{row['sign_consistency']:>10.3f}  {status:>10}"
        )

    print(f"\n{len(selected)} features selected out of {len(shap_df)} candidates")

    if selected:
        print("\n" + "=" * 65)
        print("Paste into formula_model.py FEATURES list:")
        print("=" * 65)
        print("FEATURES = [")
        for f in selected:
            row = shap_df[shap_df["feature"] == f].iloc[0]
            print(f'    "{f}",  # mean|SHAP|={row["mean_abs_shap"]:.4f}, '
                  f'consistency={row["sign_consistency"]:.2f}')
        print("]")

    out_csv = PROCESSED_DIR / "shap_selected_features.csv"
    shap_df["selected"] = shap_df["feature"].isin(selected)
    shap_df.to_csv(out_csv, index=False)
    log.info(f"\nResults saved to {out_csv}")

    plot_shap_results(shap_df, selected, PROCESSED_DIR / "shap_candidates.png")

    return selected


if __name__ == "__main__":
    selected = main()
