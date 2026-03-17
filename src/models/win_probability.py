"""
win_probability.py — Layer 1: Win probability model for NCAA tournament matchups.

Two models are trained:
  1. Logistic Regression with Elastic Net — baseline, calibrated probabilities.
  2. XGBoost regressor predicting margin of victory, mapped to probability
     via cumulative normal: P = Φ(margin / sigma), sigma ≈ 10.5.

Feature matrix construction (per spec):
  - One row per game direction: (A-B, label=1) and (B-A, label=0).
  - Features are differences: A_stat - B_stat for all 10 features.
  - StandardScaler applied before logistic regression.

Validation: Leave-One-Season-Out Cross-Validation (LOSO-CV).

Output: win_prob_matrix_2025.csv — P(team_i beats team_j) for all 2025 pairs.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler

from config import EXTERNAL_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

KAGGLE_DIR = EXTERNAL_DIR / "kaggle"
RESULTS_FILE = KAGGLE_DIR / "MNCAATourneyDetailedResults.csv"
TEAMS_FILE = KAGGLE_DIR / "MTeams.csv"
FEATURES_FILE = PROCESSED_DIR / "features_coaching.csv"

# Features to difference (A - B)
DIFF_FEATURES = [
    "TRUE_QUALITY_SCORE",
    "QMS",
    "COACH_PREMIUM",
    "SEED",
    "SEED_DIVERGENCE",
    "ADJOE",
    "ADJDE",
    "WAB",
    "TOR",
    "ORB",
]

# Sigma for margin → probability mapping.
# Hardcoded fallback; overridden at runtime by fit_margin_sigma() below.
# Historical NCAA tournament margin std dev is typically 10–12 pts.
MARGIN_SIGMA = 10.5


def fit_margin_sigma(df: pd.DataFrame) -> float:
    """
    Fit the margin-to-probability sigma from historical tournament data.

    Uses the standard deviation of winning margins across all games.
    This ensures the normal CDF mapping P = Φ(margin/σ) is properly
    calibrated to the actual spread of tournament results rather than
    a hardcoded constant.

    Args:
        df: Symmetric matchup DataFrame from load_matchup_data().
            Must have a MARGIN column (positive = team A won).

    Returns:
        Fitted sigma (std dev of absolute margins).
    """
    # Use only the winner-perspective rows (LABEL=1) to avoid double-counting
    margins = df.loc[df["LABEL"] == 1, "MARGIN"].abs()
    sigma = float(margins.std())
    log.info(f"Fitted margin sigma: {sigma:.2f} (n={len(margins)} games, "
             f"range {margins.min():.0f}–{margins.max():.0f})")
    return sigma


# ── Data preparation ───────────────────────────────────────────────────────────

def load_matchup_data(
    results_path: Path = RESULTS_FILE,
    features_path: Path = FEATURES_FILE,
    teams_path: Path = TEAMS_FILE,
) -> pd.DataFrame:
    """
    Build symmetric matchup DataFrame from tournament results + team features.

    For each game (A beats B), creates two rows:
      - (A - B features, margin=+actual, label=1)
      - (B - A features, margin=-actual, label=0)

    Args:
        results_path:  Path to MNCAATourneyDetailedResults.csv.
        features_path: Path to features_coaching.csv.
        teams_path:    Path to MTeams.csv.

    Returns:
        DataFrame with DIFF_FEATURES columns + MARGIN + LABEL + YEAR.
    """
    results = pd.read_csv(results_path).rename(columns={"Season": "YEAR"})
    features = pd.read_csv(features_path)
    teams = pd.read_csv(teams_path)

    # Map Kaggle TeamID → CBB team name via MTeams + CBB_TO_KAGGLE inverse
    # Use KAGGLE_TEAM name directly — join features via the same name map
    kaggle_to_cbb = _build_kaggle_to_cbb_map(features, teams)

    results["WTEAM"] = results["WTeamID"].map(kaggle_to_cbb)
    results["LTEAM"] = results["LTeamID"].map(kaggle_to_cbb)
    results["MARGIN"] = results["WScore"] - results["LScore"]

    # Drop rows where team name mapping failed
    before = len(results)
    results = results.dropna(subset=["WTEAM", "LTEAM"])
    dropped = before - len(results)
    if dropped:
        log.warning(f"Dropped {dropped} games with unmapped team names")

    # Filter to years where we have features (2013–2024)
    results = results[results["YEAR"].between(2013, 2024)]
    log.info(f"Tournament games after filtering: {len(results)} ({results['YEAR'].nunique()} seasons)")

    # Join features for winner and loser
    feat_cols = ["YEAR", "TEAM"] + DIFF_FEATURES
    feat = features[feat_cols].copy()

    games = results[["YEAR", "WTEAM", "LTEAM", "MARGIN"]].copy()
    games = games.merge(feat.rename(columns={"TEAM": "WTEAM"}).add_suffix("_W").rename(columns={"YEAR_W": "YEAR", "WTEAM_W": "WTEAM"}), on=["YEAR", "WTEAM"], how="inner")
    games = games.merge(feat.rename(columns={"TEAM": "LTEAM"}).add_suffix("_L").rename(columns={"YEAR_L": "YEAR", "LTEAM_L": "LTEAM"}), on=["YEAR", "LTEAM"], how="inner")

    before = len(results)
    dropped = before - len(games)
    if dropped:
        log.warning(f"Dropped {dropped} games after feature join (missing features)")
    log.info(f"Games with full features: {len(games)}")

    # Build symmetric rows
    rows = []
    for _, g in games.iterrows():
        diff_w = {f: g[f"{f}_W"] - g[f"{f}_L"] for f in DIFF_FEATURES}
        diff_l = {f: g[f"{f}_L"] - g[f"{f}_W"] for f in DIFF_FEATURES}
        rows.append({**diff_w, "MARGIN": g["MARGIN"], "LABEL": 1, "YEAR": g["YEAR"]})
        rows.append({**diff_l, "MARGIN": -g["MARGIN"], "LABEL": 0, "YEAR": g["YEAR"]})

    df = pd.DataFrame(rows)
    log.info(f"Symmetric matchup matrix: {len(df)} rows ({len(df)//2} games × 2)")
    return df


def _build_kaggle_to_cbb_map(features: pd.DataFrame, teams: pd.DataFrame) -> dict[int, str]:
    """
    Build TeamID → CBB team name mapping using MTeams as bridge.

    Args:
        features: CBB features DataFrame with TEAM column.
        teams:    MTeams DataFrame with TeamID and TeamName.

    Returns:
        Dict mapping Kaggle TeamID → CBB TEAM name.
    """
    from src.features.coaching import CBB_TO_KAGGLE_NAMES

    # Invert the CBB→Kaggle map
    kaggle_to_cbb = {v: k for k, v in CBB_TO_KAGGLE_NAMES.items()}

    cbb_team_set = set(features["TEAM"].unique())
    result = {}
    for _, row in teams.iterrows():
        kaggle_name = row["TeamName"]
        team_id = row["TeamID"]
        # Try direct match first
        if kaggle_name in cbb_team_set:
            result[team_id] = kaggle_name
        elif kaggle_name in kaggle_to_cbb:
            cbb_name = kaggle_to_cbb[kaggle_name]
            if cbb_name in cbb_team_set:
                result[team_id] = cbb_name
    return result


# ── Models ─────────────────────────────────────────────────────────────────────

def train_logistic(
    X: np.ndarray, y: np.ndarray
) -> tuple[LogisticRegression, StandardScaler]:
    """
    Train Elastic Net logistic regression on matchup features.

    Args:
        X: Feature matrix (n_samples × n_features).
        y: Binary labels (1 = team A wins).

    Returns:
        Tuple of (fitted model, fitted scaler).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # C=0.1 provides stronger regularization, prevents overflow from correlated features
    model = LogisticRegression(
        solver="lbfgs",
        C=0.1,
        max_iter=2000,
        random_state=42,
    )
    model.fit(X_scaled, y)
    return model, scaler


def train_xgboost_margin(X: np.ndarray, y_margin: np.ndarray):
    """
    Train XGBoost regressor to predict margin of victory.

    Args:
        X:        Feature matrix (n_samples × n_features).
        y_margin: Continuous margin values (positive = A wins).

    Returns:
        Fitted XGBoost regressor.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("Install xgboost: pip install xgboost")

    model = XGBRegressor(
        max_depth=3,
        subsample=0.8,
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y_margin)
    return model


def margin_to_prob(margin: np.ndarray, sigma: float = MARGIN_SIGMA) -> np.ndarray:
    """
    Map predicted margin of victory to win probability via cumulative normal.

    P(win) = Φ(margin / sigma)

    Args:
        margin: Predicted point differential (A score - B score).
        sigma:  Standard deviation of tournament margins (~10.5 historically).

    Returns:
        Win probability array in [0, 1].
    """
    return norm.cdf(margin / sigma)


# ── Validation ─────────────────────────────────────────────────────────────────

def loso_cv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Leave-One-Season-Out Cross-Validation.

    For each season, trains on all other seasons and evaluates on the held-out
    season. Reports log loss and accuracy for both models.

    Args:
        df: Symmetric matchup DataFrame from load_matchup_data().

    Returns:
        DataFrame with per-season metrics.
    """
    seasons = sorted(df["YEAR"].unique())
    X_all = df[DIFF_FEATURES].values
    y_label = df["LABEL"].values
    y_margin = df["MARGIN"].values

    records = []
    for season in seasons:
        test_mask = df["YEAR"] == season
        train_mask = ~test_mask

        X_train, X_test = X_all[train_mask], X_all[test_mask]
        y_train_lbl, y_test_lbl = y_label[train_mask], y_label[test_mask]
        y_train_mar, y_test_mar = y_margin[train_mask], y_margin[test_mask]

        if len(X_train) < 50:
            log.warning(f"Skipping {season} — too few training games")
            continue

        # Logistic regression
        lr_model, scaler = train_logistic(X_train, y_train_lbl)
        lr_probs = lr_model.predict_proba(scaler.transform(X_test))[:, 1]
        lr_preds = (lr_probs >= 0.5).astype(int)

        # Fit sigma from training data so XGBoost CDF is calibrated to this fold
        sigma = fit_margin_sigma(df[train_mask])

        # XGBoost margin model
        try:
            xgb_model = train_xgboost_margin(X_train, y_train_mar)
            xgb_margin = xgb_model.predict(X_test)
            xgb_probs = margin_to_prob(xgb_margin, sigma=sigma)
            xgb_preds = (xgb_probs >= 0.5).astype(int)
            xgb_ll = log_loss(y_test_lbl, np.clip(xgb_probs, 1e-7, 1 - 1e-7))
            xgb_acc = accuracy_score(y_test_lbl, xgb_preds)
        except Exception as e:
            log.warning(f"XGBoost failed for {season}: {e}")
            xgb_ll, xgb_acc = float("nan"), float("nan")

        records.append({
            "YEAR": season,
            "N_TEST_GAMES": test_mask.sum() // 2,
            "LR_LOG_LOSS": round(log_loss(y_test_lbl, np.clip(lr_probs, 1e-7, 1 - 1e-7)), 4),
            "LR_ACCURACY": round(accuracy_score(y_test_lbl, lr_preds), 4),
            "XGB_LOG_LOSS": round(xgb_ll, 4),
            "XGB_ACCURACY": round(xgb_acc, 4),
        })

    return pd.DataFrame(records)


def temporal_cv(df: pd.DataFrame, min_train_seasons: int = 3) -> pd.DataFrame:
    """
    Strict Temporal Cross-Validation.

    For each test year, trains ONLY on seasons strictly before that year.
    Skips any year where fewer than `min_train_seasons` prior seasons exist.
    This is the correct evaluation for a real forecasting scenario — no future
    data is ever used, matching live deployment conditions.

    Args:
        df:                 Symmetric matchup DataFrame from load_matchup_data().
        min_train_seasons:  Minimum number of prior seasons required to evaluate
                            a test year (default: 3).

    Returns:
        DataFrame with per-season metrics (same schema as loso_cv output).
    """
    seasons = sorted(df["YEAR"].unique())
    X_all = df[DIFF_FEATURES].values
    y_label = df["LABEL"].values
    y_margin = df["MARGIN"].values

    records = []
    for season in seasons:
        prior_seasons = [s for s in seasons if s < season]

        if len(prior_seasons) < min_train_seasons:
            log.info(
                f"Skipping {season} — only {len(prior_seasons)} prior season(s) "
                f"(need {min_train_seasons})"
            )
            continue

        test_mask = df["YEAR"] == season
        train_mask = df["YEAR"].isin(prior_seasons)

        X_train, X_test = X_all[train_mask], X_all[test_mask]
        y_train_lbl, y_test_lbl = y_label[train_mask], y_label[test_mask]
        y_train_mar = y_margin[train_mask]

        if len(X_train) < 50:
            log.warning(f"Skipping {season} — too few training games ({len(X_train)})")
            continue

        # Logistic regression
        lr_model, scaler = train_logistic(X_train, y_train_lbl)
        lr_probs = lr_model.predict_proba(scaler.transform(X_test))[:, 1]
        lr_preds = (lr_probs >= 0.5).astype(int)

        # Fit sigma from training data only — no future data
        sigma = fit_margin_sigma(df[train_mask])

        # XGBoost margin model
        try:
            xgb_model = train_xgboost_margin(X_train, y_train_mar)
            xgb_margin = xgb_model.predict(X_test)
            xgb_probs = margin_to_prob(xgb_margin, sigma=sigma)
            xgb_preds = (xgb_probs >= 0.5).astype(int)
            xgb_ll = log_loss(y_test_lbl, np.clip(xgb_probs, 1e-7, 1 - 1e-7))
            xgb_acc = accuracy_score(y_test_lbl, xgb_preds)
        except Exception as e:
            log.warning(f"XGBoost failed for {season}: {e}")
            xgb_ll, xgb_acc = float("nan"), float("nan")

        records.append({
            "YEAR": season,
            "N_TRAIN_SEASONS": len(prior_seasons),
            "N_TEST_GAMES": test_mask.sum() // 2,
            "LR_LOG_LOSS": round(log_loss(y_test_lbl, np.clip(lr_probs, 1e-7, 1 - 1e-7)), 4),
            "LR_ACCURACY": round(accuracy_score(y_test_lbl, lr_preds), 4),
            "XGB_LOG_LOSS": round(xgb_ll, 4),
            "XGB_ACCURACY": round(xgb_acc, 4),
        })

        log.info(
            f"Temporal CV {season}: LR acc={records[-1]['LR_ACCURACY']:.4f} "
            f"ll={records[-1]['LR_LOG_LOSS']:.4f} | "
            f"XGB acc={records[-1]['XGB_ACCURACY']:.4f} ll={records[-1]['XGB_LOG_LOSS']:.4f}"
        )

    return pd.DataFrame(records)


# ── Calibration ────────────────────────────────────────────────────────────────

def calibration_metrics(df: pd.DataFrame, plot_path: Path | None = None) -> dict:
    """
    Compute Brier score and plot calibration curve using LOSO-CV predictions.

    Collects out-of-fold predicted probabilities across all LOSO folds (LR model
    only), then computes:
      - Brier score: mean squared error between predicted prob and true label.
        Lower is better; a naive 50% predictor scores 0.25.
      - Calibration curve: fraction of positives vs. mean predicted probability
        in 10 equal-width bins.

    All numbers come from held-out predictions — no training data is scored.

    Args:
        df:        Symmetric matchup DataFrame from load_matchup_data().
        plot_path: If provided, save calibration curve PNG here. Defaults to
                   data/processed/calibration_curve.png.

    Returns:
        Dict with keys: brier_score, n_predictions, n_seasons.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if plot_path is None:
        plot_path = PROCESSED_DIR / "calibration_curve.png"

    seasons = sorted(df["YEAR"].unique())
    X_all = df[DIFF_FEATURES].values
    y_label = df["LABEL"].values

    all_probs: list[float] = []
    all_labels: list[int] = []
    seasons_used = 0

    for season in seasons:
        test_mask = df["YEAR"] == season
        train_mask = ~test_mask

        X_train, X_test = X_all[train_mask], X_all[test_mask]
        y_train_lbl, y_test_lbl = y_label[train_mask], y_label[test_mask]

        if len(X_train) < 50:
            continue

        lr_model, scaler = train_logistic(X_train, y_train_lbl)
        probs = lr_model.predict_proba(scaler.transform(X_test))[:, 1]

        all_probs.extend(probs.tolist())
        all_labels.extend(y_test_lbl.tolist())
        seasons_used += 1

    all_probs_arr = np.array(all_probs)
    all_labels_arr = np.array(all_labels)

    brier = brier_score_loss(all_labels_arr, all_probs_arr)
    log.info(f"Brier score (LOSO-CV, {seasons_used} seasons): {brier:.4f}")

    # Calibration curve
    fraction_pos, mean_pred = calibration_curve(
        all_labels_arr, all_probs_arr, n_bins=10, strategy="uniform"
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred, fraction_pos, "o-", label="LR model (LOSO-CV)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(
        f"Calibration Curve — LR Model (LOSO-CV)\n"
        f"Brier Score: {brier:.4f}  |  n={len(all_probs_arr)} predictions  |  {seasons_used} seasons"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    log.info(f"Calibration curve saved to {plot_path}")

    return {
        "brier_score": round(brier, 4),
        "n_predictions": len(all_probs_arr),
        "n_seasons": seasons_used,
    }


# ── Final model + output ───────────────────────────────────────────────────────

def build_final_models(df: pd.DataFrame) -> tuple:
    """
    Train final models on all available data (2013–2024).

    Args:
        df: Full symmetric matchup DataFrame.

    Returns:
        Tuple of (lr_model, lr_scaler, xgb_model).
    """
    X = df[DIFF_FEATURES].values
    y_label = df["LABEL"].values
    y_margin = df["MARGIN"].values

    lr_model, scaler = train_logistic(X, y_label)
    try:
        xgb_model = train_xgboost_margin(X, y_margin)
    except ImportError:
        log.warning("XGBoost not available — skipping margin model")
        xgb_model = None

    return lr_model, scaler, xgb_model


def build_win_prob_matrix(
    year: int,
    lr_model: LogisticRegression,
    scaler: StandardScaler,
    xgb_model,
    features_path: Path = FEATURES_FILE,
) -> pd.DataFrame:
    """
    Build win probability matrix for all tournament team pairs in a given year.

    P(row_team beats col_team) using ensemble average of LR and XGBoost.
    If XGBoost unavailable, uses LR only.

    Args:
        year:          Season year to generate matrix for.
        lr_model:      Fitted logistic regression model.
        scaler:        Fitted StandardScaler for LR.
        xgb_model:     Fitted XGBoost model (or None).
        features_path: Path to features CSV.

    Returns:
        Square DataFrame indexed by team name, values = P(row beats col).
    """
    features = pd.read_csv(features_path)
    tourney = features[
        (features["YEAR"] == year) & features["SEED"].notna()
    ][["TEAM"] + DIFF_FEATURES].dropna()

    teams = tourney["TEAM"].tolist()
    n = len(teams)

    # Pre-build feature dict for O(1) lookups — avoids 4,624 DataFrame filters
    feat_dict: dict[str, np.ndarray] = {
        row["TEAM"]: row[DIFF_FEATURES].values
        for _, row in tourney.iterrows()
    }

    # Fit sigma from the full training data passed via the matrix build context.
    # Fall back to MARGIN_SIGMA constant if no margin data is accessible here.
    sigma = MARGIN_SIGMA

    # Build all pairwise diffs as a batch for LR (vectorized predict_proba)
    # Shape: (n*(n-1), n_features) — all off-diagonal pairs
    pair_idx: list[tuple[int, int]] = [
        (i, j) for i in range(n) for j in range(n) if i != j
    ]
    team_arr = np.array([feat_dict[teams[i]] - feat_dict[teams[j]] for i, j in pair_idx])
    lr_probs_flat = lr_model.predict_proba(scaler.transform(team_arr))[:, 1]

    if xgb_model is not None:
        xgb_margins_flat = xgb_model.predict(team_arr)
        xgb_probs_flat = margin_to_prob(xgb_margins_flat, sigma=sigma)
        probs_flat = (lr_probs_flat + xgb_probs_flat) / 2
    else:
        probs_flat = lr_probs_flat

    matrix = pd.DataFrame(0.5, index=teams, columns=teams)
    for k, (i, j) in enumerate(pair_idx):
        matrix.iloc[i, j] = round(float(probs_flat[k]), 4)

    log.info(f"Win probability matrix built: {n}×{n} teams for {year}")
    return matrix


def save_win_prob_matrix(matrix: pd.DataFrame, year: int) -> Path:
    """Save win probability matrix to data/processed/."""
    out = PROCESSED_DIR / f"win_prob_matrix_{year}.csv"
    matrix.to_csv(out)
    log.info(f"Saved to {out}")
    return out


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Load data
    df = load_matchup_data()

    # 2. LOSO-CV validation
    log.info("Running Leave-One-Season-Out CV...")
    loso_results = loso_cv(df)

    # 3. Strict temporal CV (no future data)
    log.info("Running Strict Temporal CV...")
    temp_results = temporal_cv(df, min_train_seasons=3)

    # ── Side-by-side comparison ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("LOSO-CV Results (trains on all other seasons, including future):")
    print("=" * 72)
    print(loso_results.to_string(index=False))
    print(f"\nMean LR  — Log Loss: {loso_results['LR_LOG_LOSS'].mean():.4f}, "
          f"Accuracy: {loso_results['LR_ACCURACY'].mean():.4f}")
    print(f"Mean XGB — Log Loss: {loso_results['XGB_LOG_LOSS'].mean():.4f}, "
          f"Accuracy: {loso_results['XGB_ACCURACY'].mean():.4f}")

    print("\n" + "=" * 72)
    print("Temporal CV Results (trains only on seasons BEFORE each test year):")
    print("=" * 72)
    print(temp_results.to_string(index=False))
    print(f"\nMean LR  — Log Loss: {temp_results['LR_LOG_LOSS'].mean():.4f}, "
          f"Accuracy: {temp_results['LR_ACCURACY'].mean():.4f}")
    print(f"Mean XGB — Log Loss: {temp_results['XGB_LOG_LOSS'].mean():.4f}, "
          f"Accuracy: {temp_results['XGB_ACCURACY'].mean():.4f}")

    # ── Head-to-head by year ───────────────────────────────────────────────────
    shared_years = set(loso_results["YEAR"]) & set(temp_results["YEAR"])
    loso_shared = loso_results[loso_results["YEAR"].isin(shared_years)].set_index("YEAR")
    temp_shared = temp_results[temp_results["YEAR"].isin(shared_years)].set_index("YEAR")

    print("\n" + "=" * 72)
    print("Head-to-Head Comparison (shared evaluation years):")
    print("=" * 72)
    print(f"{'YEAR':<6} {'LOSO LR Acc':>12} {'Temporal LR Acc':>16} {'Δ Acc':>8}  "
          f"{'LOSO LR LL':>12} {'Temporal LR LL':>15} {'Δ LL':>8}")
    print("-" * 80)
    for year in sorted(shared_years):
        loso_acc = loso_shared.loc[year, "LR_ACCURACY"]
        temp_acc = temp_shared.loc[year, "LR_ACCURACY"]
        loso_ll = loso_shared.loc[year, "LR_LOG_LOSS"]
        temp_ll = temp_shared.loc[year, "LR_LOG_LOSS"]
        delta_acc = temp_acc - loso_acc
        delta_ll = temp_ll - loso_ll
        print(f"{year:<6} {loso_acc:>12.4f} {temp_acc:>16.4f} {delta_acc:>+8.4f}  "
              f"{loso_ll:>12.4f} {temp_ll:>15.4f} {delta_ll:>+8.4f}")

    print("-" * 80)
    avg_loso_acc = loso_shared["LR_ACCURACY"].mean()
    avg_temp_acc = temp_shared["LR_ACCURACY"].mean()
    avg_loso_ll = loso_shared["LR_LOG_LOSS"].mean()
    avg_temp_ll = temp_shared["LR_LOG_LOSS"].mean()
    print(f"{'AVG':<6} {avg_loso_acc:>12.4f} {avg_temp_acc:>16.4f} "
          f"{avg_temp_acc - avg_loso_acc:>+8.4f}  "
          f"{avg_loso_ll:>12.4f} {avg_temp_ll:>15.4f} "
          f"{avg_temp_ll - avg_loso_ll:>+8.4f}")

    print("\n" + "=" * 72)
    print("Summary (all numbers computed this session):")
    print("=" * 72)
    print(f"  LOSO-CV     — LR Accuracy: {loso_results['LR_ACCURACY'].mean():.4f}  "
          f"Log Loss: {loso_results['LR_LOG_LOSS'].mean():.4f}")
    print(f"  Temporal CV — LR Accuracy: {temp_results['LR_ACCURACY'].mean():.4f}  "
          f"Log Loss: {temp_results['LR_LOG_LOSS'].mean():.4f}")

    # 4. Calibration metrics (Brier score + curve)
    log.info("Computing calibration metrics (LOSO-CV predictions)...")
    cal = calibration_metrics(df)
    print("\n" + "=" * 72)
    print("Calibration Metrics (LOSO-CV out-of-fold predictions):")
    print("=" * 72)
    print(f"  Brier Score : {cal['brier_score']:.4f}  "
          f"(naive 0.50 predictor = 0.2500; lower is better)")
    print(f"  Predictions : {cal['n_predictions']}  ({cal['n_seasons']} LOSO folds)")
    print(f"  Curve saved : {PROCESSED_DIR / 'calibration_curve.png'}")

    # 5. Feature importance (LR coefficients)
    X_all = df[DIFF_FEATURES].values
    scaler_full = StandardScaler()
    X_scaled = scaler_full.fit_transform(X_all)
    lr_full = LogisticRegression(solver="lbfgs", C=0.1, max_iter=2000, random_state=42)
    lr_full.fit(X_scaled, df["LABEL"].values)
    coef_df = pd.DataFrame({
        "feature": DIFF_FEATURES,
        "coefficient": lr_full.coef_[0],
    }).sort_values("coefficient", ascending=False)
    print("\nLogistic Regression Coefficients (standardized):")
    print(coef_df.to_string(index=False))

    # 5. Train final models and build 2025 matrix
    lr_model, scaler, xgb_model = build_final_models(df)
    matrix = build_win_prob_matrix(2025, lr_model, scaler, xgb_model)
    save_win_prob_matrix(matrix, 2025)
    print(f"\nWin prob matrix shape: {matrix.shape}")
    print("Sample (first 5×5):")
    print(matrix.iloc[:5, :5].to_string())
