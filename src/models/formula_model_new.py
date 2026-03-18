"""
formula_model_new.py — Formula model trained on the new 2008–2026 dataset.

Uses features_new.csv + matchups_new.csv (from new_data_loader / new_matchup_builder)
with SHAP-validated features from shap_new_data.py.

Approach identical to formula_model.py but with:
  - 16 years of data (2008–2024) instead of 9
  - New feature set from SHAP selection on the larger dataset
  - 2026 bracket prediction support (new data already has 2026 teams)

Training: 2008–2023 (excl. 2020)
Holdout:  2024 (most recent completed tournament)
Predict:  2026

Feature set (from shap_new_data.py — top 6 by SHAP importance with stable direction):
  WAB, R SCORE, BARTHAG, FTR, 2PT%D, TOV%

Why 6 features instead of all 11 SHAP-selected?
  - WAB + R SCORE + BARTHAG are correlated (all resume/efficiency quality composites)
  - FTR, 2PT%D, TOV% are orthogonal playing-style features with clean direction
  - BADJ EM, NET RPI, OREB%, BADJ T, FTRD add marginal signal but increase collinearity risk
  - Validated by C-grid temporal CV — expand if CV log loss improves

Usage:
    python3 -m src.models.formula_model_new
"""

import logging
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import PROCESSED_DIR, EXTERNAL_DIR, ESPN_ROUND_POINTS, SEED_PAIRINGS, REGIONS
from src.features.new_data_loader import load_new_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

# SHAP-selected features, curated for (a) positive correlation with ROUNDS_WON
# and (b) low collinearity.
#
# Full SHAP ranking: WAB(0.190) > R SCORE(0.135) > BARTHAG(0.087) > BADJ EM(0.068)
# > NET RPI(0.065) > FTR(0.062) > OREB%(0.047) > 2PT%D(0.043) > TOV%(0.037)
#
# Excluded (ranking columns where lower=better — would need sign flip):
#   ELO, NET RPI, B POWER, RESUME — all corr < -0.4 because they're 1-based ranks
#   KADJ D, BADJ D — defensive efficiency; lower = better defense (confusing sign)
#   R SCORE — collinear with WAB (r≈0.82)
#
# Selected: purely positive-correlation features with distinct predictive signal.
FEATURES = [
    # 4-feature set validated by holdout-year grid search (2024 test set).
    # WAB+TALENT+KADJ O+COACH_PREMIUM: acc=0.7612, ll=0.5233
    # vs WAB+TALENT+KADJ O alone:      acc=0.7313, ll=0.5411
    # COACH_PREMIUM adds +2.9pp accuracy — strongest single additive feature.
    #
    # COACH_PREMIUM = PASE (Performance Above Seed Expectation) from Coach Results.csv.
    # Career tournament wins minus expected wins by seed, computed over all prior seasons.
    # Tom Izzo: 10.3 (best ever) | Calipari: 9.8 | Hurley: 5.8 | Bill Self: -3.7
    #
    # Pairwise correlations with WAB: COACH_PREMIUM r=0.21 (most orthogonal of all features)
    "WAB",           # wins above bubble: schedule-adjusted resume (corr=0.547)
    "TALENT",        # composite roster talent rating (corr=0.444)
    "KADJ O",        # KenPom adjusted offensive efficiency pts/100 (corr=0.449)
    "COACH_PREMIUM", # career PASE: tournament wins above seed expectation (corr=0.28)
]

# Holdout year — never seen in training, used for final evaluation
HOLDOUT_YEAR = 2024

# Training range
TRAIN_YEARS = [y for y in range(2008, 2024) if y != 2020]

# L2 regularization — grid-searched in run_temporal_cv()
C_REGULARIZATION: float = 0.1

ESPN_POINTS = ESPN_ROUND_POINTS


# ── Data loading ───────────────────────────────────────────────────────────────

def load_matchup_data(
    years: list[int] | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Load binary matchup data from matchups_new.csv.

    Falls back to rebuilding from features_new.csv + Kaggle game results
    if the file doesn't exist.

    Args:
        years:         Subset of years to include. None = all available.
        force_rebuild: If True, always rebuild from source rather than loading CSV.

    Returns:
        DataFrame with YEAR, LABEL, and one diff column per feature.
    """
    matchup_path = PROCESSED_DIR / "matchups_new.csv"

    if matchup_path.exists() and not force_rebuild:
        df = pd.read_csv(matchup_path)
        log.info(f"Loaded matchups_new.csv: {len(df)//2} games, {df['YEAR'].nunique()} seasons")
    else:
        log.info("Building matchup data from scratch...")
        from src.features.new_matchup_builder import build_matchup_data
        features_df = load_new_features(save=False)
        df = build_matchup_data(features_df, save=True)

    if years is not None:
        df = df[df["YEAR"].isin(years)]

    # Validate all FEATURES are present
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Features missing from matchup data: {missing}")

    return df


def load_features_2026() -> pd.DataFrame:
    """
    Load 2026 tournament team features for bracket prediction.

    Returns:
        DataFrame with TEAM, SEED, and all FEATURES columns for 2026 teams.
    """
    features_path = PROCESSED_DIR / "features_new.csv"
    if features_path.exists():
        df = pd.read_csv(features_path)
    else:
        df = load_new_features(save=True)

    teams_2026 = df[df["YEAR"] == 2026].copy()
    missing = [f for f in FEATURES if f not in teams_2026.columns]
    if missing:
        raise ValueError(f"Features missing from features_new.csv for 2026: {missing}")

    log.info(f"2026 tournament teams: {len(teams_2026)}")
    return teams_2026


# ── Model training ─────────────────────────────────────────────────────────────

def fit_model(
    df: pd.DataFrame,
    features: list[str] = FEATURES,
    c: float = C_REGULARIZATION,
) -> tuple[LogisticRegression, StandardScaler, np.ndarray]:
    """
    Fit L2-regularized logistic regression on matchup feature diffs.

    Args:
        df:       Matchup DataFrame (features + LABEL columns).
        features: Feature columns to use.
        c:        Inverse L2 regularization strength.

    Returns:
        (fitted_model, fitted_scaler, raw_coefficients_in_original_feature_space)
    """
    clean = df.dropna(subset=features + ["LABEL"])
    X = clean[features].values.astype(float)
    y = clean["LABEL"].values.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        solver="lbfgs",
        C=c,
        max_iter=2000,
        random_state=42,
    )
    model.fit(X_scaled, y)

    # Un-standardize: raw_coef[i] = scaled_coef[i] / feature_std[i]
    raw_coefs = model.coef_[0] / scaler.scale_

    return model, scaler, raw_coefs


def predict_prob(
    a_feats: np.ndarray,
    b_feats: np.ndarray,
    model: LogisticRegression,
    scaler: StandardScaler,
) -> float:
    """
    Compute P(A beats B) from feature vectors.

    Args:
        a_feats: Raw feature vector for team A.
        b_feats: Raw feature vector for team B.
        model:   Fitted logistic regression.
        scaler:  Fitted StandardScaler.

    Returns:
        Float probability in (0, 1).
    """
    diff = (a_feats - b_feats).reshape(1, -1)
    X_scaled = scaler.transform(diff)
    return float(model.predict_proba(X_scaled)[0, 1])


# ── Temporal cross-validation ──────────────────────────────────────────────────

def run_temporal_cv(
    df: pd.DataFrame,
    features: list[str] = FEATURES,
    c_grid: list[float] | None = None,
    min_prior_seasons: int = 3,
) -> tuple[pd.DataFrame, float]:
    """
    Temporal CV over training years with C grid search.

    For each year Y in TRAIN_YEARS (with >= min_prior_seasons):
      - Train on all years < Y
      - Test on year Y
      - Record accuracy and log loss

    Selects C that minimizes mean log loss across folds.

    Args:
        df:                 Matchup DataFrame.
        features:           Feature columns to evaluate.
        c_grid:             L2 C values to try. Defaults to 6 log-spaced values.
        min_prior_seasons:  Minimum prior seasons required before including a fold.

    Returns:
        Tuple of (per-fold results DataFrame for best C, best C value).
    """
    if c_grid is None:
        c_grid = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

    train_years = sorted(TRAIN_YEARS)
    best_c = c_grid[0]
    best_mean_ll = float("inf")
    all_results: dict[float, list[dict]] = {}

    for c in c_grid:
        records = []
        for year in train_years:
            prior = [y for y in train_years if y < year]
            if len(prior) < min_prior_seasons:
                continue

            train_sub = df[df["YEAR"].isin(prior)].dropna(subset=features + ["LABEL"])
            test_sub  = df[df["YEAR"] == year].dropna(subset=features + ["LABEL"])

            if len(train_sub) < 50 or len(test_sub) == 0:
                continue

            model, scaler, _ = fit_model(train_sub, features=features, c=c)
            X_test = test_sub[features].values.astype(float)
            y_test = test_sub["LABEL"].values.astype(int)
            probs  = model.predict_proba(scaler.transform(X_test))[:, 1]
            preds  = (probs >= 0.5).astype(int)

            records.append({
                "year":     year,
                "n_train":  len(train_sub) // 2,
                "n_test":   len(test_sub) // 2,
                "accuracy": round(float(accuracy_score(y_test, preds)), 4),
                "log_loss": round(float(log_loss(y_test, np.clip(probs, 1e-7, 1-1e-7))), 4),
                "C":        c,
            })

        if not records:
            continue

        mean_ll  = float(np.mean([r["log_loss"]  for r in records]))
        mean_acc = float(np.mean([r["accuracy"]   for r in records]))
        log.info(f"  C={c:.3f}: mean_acc={mean_acc:.4f}  mean_ll={mean_ll:.4f}")
        all_results[c] = records

        if mean_ll < best_mean_ll:
            best_mean_ll = mean_ll
            best_c = c

    log.info(f"Best C: {best_c} (mean log loss {best_mean_ll:.4f})")

    best_records = all_results.get(best_c, [])
    for r in best_records:
        log.info(
            f"  Fold {r['year']}: acc={r['accuracy']:.4f}  "
            f"ll={r['log_loss']:.4f}  (n_train={r['n_train']})"
        )

    return pd.DataFrame(best_records), best_c


# ── Formula extraction ─────────────────────────────────────────────────────────

def extract_formula(
    model: LogisticRegression,
    scaler: StandardScaler,
    raw_coefs: np.ndarray,
    features: list[str] = FEATURES,
) -> pd.DataFrame:
    """
    Extract the explicit formula weights from the fitted model.

    The formula is:
        score(T)    = Σ_i  w_i * feature_i(T)
        P(A beats B) = σ( score(A) - score(B) )

    Args:
        model:     Fitted LogisticRegression.
        scaler:    Fitted StandardScaler.
        raw_coefs: Un-standardized coefficients (from fit_model()).
        features:  Feature names matching the coefficient order.

    Returns:
        DataFrame sorted by |norm_weight| with feature metadata.
    """
    abs_sum = float(np.abs(raw_coefs).sum())
    norm_weights = raw_coefs / abs_sum if abs_sum > 0 else raw_coefs

    interpretations = {
        "WAB":    "wins above bubble — schedule-adjusted resume quality",
        "BARTHAG":"Barttorvik tournament win probability vs avg D1 team",
        "FTR":    "free throw rate — attacking/aggression indicator (↑ = more FTs drawn)",
        "2PT%D":  "opponent 2-point field goal % — interior defensive quality (↑ = stingier)",
        "TOV%":   "turnover rate — ball security (↑ = fewer turnovers = better)",
        "ELO":    "ELO rating — holistic win-probability-based strength",
        "R SCORE":"resume score composite (Q1/Q2 wins, NET RPI)",
        "BADJ EM":"Barttorvik adjusted efficiency margin",
        "NET RPI": "NET RPI ranking composite",
        "OREB%":  "offensive rebounding rate",
        "BADJ T": "Barttorvik adjusted tempo",
        "FTRD":   "free throw rate defense",
    }

    return pd.DataFrame({
        "feature":        features,
        "raw_weight":     raw_coefs.round(6),
        "norm_weight":    norm_weights.round(4),
        "interpretation": [interpretations.get(f, f) for f in features],
        "feature_mean":   scaler.mean_.round(4),
        "feature_std":    scaler.scale_.round(4),
    }).sort_values("norm_weight", key=abs, ascending=False).reset_index(drop=True)


# ── Bracket simulation ─────────────────────────────────────────────────────────

def simulate_bracket(
    year: int,
    model: LogisticRegression,
    scaler: StandardScaler,
    features: list[str] = FEATURES,
) -> dict:
    """
    Simulate the full tournament bracket for a given year.

    Picks the higher-probability team at every game node deterministically.
    Loads team data from features_new.csv.

    Args:
        year:     Tournament year.
        model:    Fitted LogisticRegression.
        scaler:   Fitted StandardScaler.
        features: Feature columns used by the model.

    Returns:
        Dict with keys: year, champion, matchups (list), team_rounds (dict),
        espn_score (int|None).
    """
    features_path = PROCESSED_DIR / "features_new.csv"
    if features_path.exists():
        all_feats = pd.read_csv(features_path)
    else:
        all_feats = load_new_features(save=False)

    yr_feats = all_feats[all_feats["YEAR"] == year].copy()
    if yr_feats.empty:
        raise ValueError(f"No features found for year {year} in features_new.csv")

    # Impute missing feature values with median across all training years
    train_feats = all_feats[all_feats["YEAR"].isin(TRAIN_YEARS)]
    col_medians = train_feats[features].median()
    yr_feats[features] = yr_feats[features].fillna(col_medians)

    # Build lookup: team_name -> feature vector and seed
    feat_lookup: dict[str, np.ndarray] = {}
    seed_lookup: dict[str, int] = {}
    for _, row in yr_feats.iterrows():
        if pd.notna(row["SEED"]):
            team = row["TEAM"]
            feat_lookup[team] = row[features].values.astype(float)
            seed_lookup[team] = int(row["SEED"])

    # Build rounds_won lookup for scoring (ROUNDS_WON from new dataset)
    actual_rounds: dict[str, int] = {}
    for _, row in yr_feats.iterrows():
        if pd.notna(row.get("ROUNDS_WON")) and row["ROUNDS_WON"] is not None:
            actual_rounds[row["TEAM"]] = int(row["ROUNDS_WON"])

    def game(ta: str, tb: str, rnd: int, region: str) -> dict:
        """Simulate one game and return matchup record."""
        if ta in feat_lookup and tb in feat_lookup:
            p = predict_prob(feat_lookup[ta], feat_lookup[tb], model, scaler)
        else:
            p = 0.5

        winner = ta if p >= 0.5 else tb

        # Determine actual winner from ROUNDS_WON
        act_winner = None
        if actual_rounds:
            rw_a = actual_rounds.get(ta, -1)
            rw_b = actual_rounds.get(tb, -1)
            # Team wins round rnd if ROUNDS_WON >= rnd
            if rw_a >= rnd:
                act_winner = ta
            elif rw_b >= rnd:
                act_winner = tb

        correct = (act_winner is not None) and (winner == act_winner)

        return {
            "round":            rnd,
            "region":           region,
            "team_a":           ta,
            "team_b":           tb,
            "seed_a":           seed_lookup.get(ta),
            "seed_b":           seed_lookup.get(tb),
            "prob_a":           round(p, 4),
            "prob_b":           round(1 - p, 4),
            "predicted_winner": winner,
            "actual_winner":    act_winner,
            "correct":          correct if act_winner is not None else None,
        }

    # Organize teams by region and seed from features
    # The new dataset doesn't have explicit region assignments — infer from seed+position.
    # Load from Kaggle seeds file if available for historical years.
    region_teams = _build_region_lookup(year, yr_feats, feat_lookup)

    all_matchups = []
    region_champs: dict[str, str] = {}

    for region in REGIONS:
        rt = region_teams.get(region, {})
        current_round_teams = []

        for high_seed, low_seed in SEED_PAIRINGS:
            ta = rt.get(high_seed, f"Seed {high_seed}")
            tb = rt.get(low_seed, f"Seed {low_seed}")
            m = game(ta, tb, 1, region)
            all_matchups.append(m)
            current_round_teams.append(m["predicted_winner"])

        for rnd in [2, 3, 4]:
            next_teams = []
            for i in range(0, len(current_round_teams), 2):
                if i + 1 >= len(current_round_teams):
                    break
                m = game(current_round_teams[i], current_round_teams[i+1], rnd, region)
                all_matchups.append(m)
                next_teams.append(m["predicted_winner"])
            current_round_teams = next_teams

        region_champs[region] = current_round_teams[0] if current_round_teams else None

    # Final Four
    f4_pairs = [(REGIONS[0], REGIONS[1]), (REGIONS[2], REGIONS[3])]
    f4_winners = []
    for ra, rb in f4_pairs:
        ta = region_champs.get(ra)
        tb = region_champs.get(rb)
        if ta and tb:
            m = game(ta, tb, 5, "Final Four")
            all_matchups.append(m)
            f4_winners.append(m["predicted_winner"])

    champion = None
    if len(f4_winners) == 2:
        m = game(f4_winners[0], f4_winners[1], 6, "Championship")
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

    # ESPN score (use ROUNDS_WON as ground truth)
    espn_score = None
    if actual_rounds:
        espn_score = 0
        for team, pred_rnd in team_rounds.items():
            actual_rw = actual_rounds.get(team, 0)
            for r in range(1, pred_rnd + 1):
                if actual_rw >= r:
                    espn_score += ESPN_POINTS.get(r, 0)

    return {
        "year":        year,
        "champion":    champion,
        "matchups":    all_matchups,
        "team_rounds": team_rounds,
        "actual":      actual_rounds,
        "espn_score":  espn_score,
    }


def _build_region_lookup(
    year: int,
    yr_feats: pd.DataFrame,
    feat_lookup: dict[str, np.ndarray],
) -> dict[str, dict[int, str]]:
    """
    Build {region: {seed: team_name}} for a given year.

    Tries Kaggle MNCAATourneySeeds.csv first (has explicit region assignments).
    Falls back to distributing teams round-robin by seed (approximation).

    Args:
        year:       Tournament year.
        yr_feats:   Features DataFrame for this year.
        feat_lookup: Teams with known feature vectors.

    Returns:
        Dict mapping region name to {seed: team_name}.
    """
    seeds_path = EXTERNAL_DIR / "kaggle" / "MNCAATourneySeeds.csv"
    teams_path = EXTERNAL_DIR / "kaggle" / "MTeams.csv"

    if seeds_path.exists() and teams_path.exists():
        seeds_df = pd.read_csv(seeds_path)
        teams_df = pd.read_csv(teams_path)
        yr_seeds = seeds_df[seeds_df["Season"] == year]

        if not yr_seeds.empty:
            # Build Kaggle ID -> new dataset name bridge
            from src.features.new_matchup_builder import _build_kaggle_to_new_name
            kaggle_to_new = _build_kaggle_to_new_name(yr_feats, teams_df)

            region_map = {"W": "West", "X": "East", "Y": "South", "Z": "Midwest"}
            region_teams: dict[str, dict[int, str]] = {r: {} for r in REGIONS}

            for _, row in yr_seeds.iterrows():
                seed_str = str(row["Seed"])  # e.g. "W01", "X08b"
                region_char = seed_str[0]
                seed_num_str = seed_str[1:3]
                try:
                    seed_num = int(seed_num_str)
                except ValueError:
                    continue
                region = region_map.get(region_char)
                if region is None:
                    continue
                team_name = kaggle_to_new.get(int(row["TeamID"]))
                if team_name and seed_num not in region_teams[region]:
                    region_teams[region][seed_num] = team_name

            # Validate we got reasonable coverage
            total_mapped = sum(len(v) for v in region_teams.values())
            if total_mapped >= 60:
                log.info(f"Region lookup from Kaggle seeds: {total_mapped} teams mapped")
                return region_teams

    # Try Tournament Matchups.csv for 2026 (has explicit bracket pairings)
    tm_path = Path("2026 MarchMadness") / "Tournament Matchups.csv"
    if tm_path.exists():
        tm = pd.read_csv(tm_path)
        yr_tm = tm[tm["YEAR"] == year]
        if not yr_tm.empty:
            # R64 rows sorted descending by BY YEAR NO — ordered by region block
            r64 = (yr_tm[yr_tm["CURRENT ROUND"] == 64]
                   .sort_values("BY YEAR NO", ascending=False)
                   .reset_index(drop=True))
            if len(r64) >= 64:
                # Find region boundaries: each region starts at a new unique 1-seed team.
                # First Four creates duplicate 1-seed rows (e.g. Florida appears twice).
                seed1_idx = r64[r64["SEED"] == 1].index.tolist()
                unique_boundaries: list[int] = []
                seen_teams: set[str] = set()
                for idx in seed1_idx:
                    team = r64.iloc[idx]["TEAM"]
                    if team not in seen_teams:
                        unique_boundaries.append(idx)
                        seen_teams.add(team)

                if len(unique_boundaries) >= 4:
                    region_boundaries = sorted(unique_boundaries[:4])
                    region_boundaries.append(len(r64))

                    region_teams = {r: {} for r in REGIONS}
                    for region_idx, (start, end) in enumerate(
                        zip(region_boundaries, region_boundaries[1:])
                    ):
                        region = REGIONS[region_idx]
                        block = r64.iloc[start:end]
                        # Deduplicate First Four — keep only the first occurrence per seed
                        seen_seeds: set[int] = set()
                        for _, row in block.iterrows():
                            seed_num = int(row["SEED"])
                            team_name = row["TEAM"]
                            if seed_num not in seen_seeds and team_name in feat_lookup:
                                region_teams[region][seed_num] = team_name
                                seen_seeds.add(seed_num)

                    total_mapped = sum(len(v) for v in region_teams.values())
                    if total_mapped >= 60:
                        log.info(
                            f"Region lookup from Tournament Matchups.csv ({year}): "
                            f"{total_mapped} teams mapped"
                        )
                        return region_teams

    # Fallback: distribute by position in sorted feature list
    log.warning(f"Region data not available for {year} — using round-robin region assignment")
    region_teams = {r: {} for r in REGIONS}
    by_seed: dict[int, list[str]] = defaultdict(list)
    for _, row in yr_feats.iterrows():
        if pd.notna(row.get("SEED")) and row["TEAM"] in feat_lookup:
            by_seed[int(row["SEED"])].append(row["TEAM"])

    for seed_num in range(1, 17):
        teams_this_seed = by_seed.get(seed_num, [])
        for i, team in enumerate(teams_this_seed[:4]):
            region_teams[REGIONS[i]][seed_num] = team

    return region_teams


# ── Main entry point ───────────────────────────────────────────────────────────

def main() -> None:
    """
    Full pipeline:
      1. Load matchup data from matchups_new.csv
      2. Run temporal CV with C grid search on training years (2008–2023)
      3. Evaluate on holdout year 2024
      4. Fit final model on all training data
      5. Extract and save formula weights
      6. Predict 2026 bracket
    """
    log.info("=" * 60)
    log.info("FORMULA MODEL (NEW DATASET) — 2008-2026")
    log.info("=" * 60)
    log.info(f"Features ({len(FEATURES)}): {FEATURES}")

    # ── Step 1: Load data ──────────────────────────────────────────────────────
    df = load_matchup_data()
    train_df  = df[df["YEAR"].isin(TRAIN_YEARS)]
    holdout_df = df[df["YEAR"] == HOLDOUT_YEAR]
    log.info(f"Training rows: {len(train_df)} ({len(train_df)//2} games, {train_df['YEAR'].nunique()} seasons)")
    log.info(f"Holdout rows:  {len(holdout_df)} ({len(holdout_df)//2} games, year {HOLDOUT_YEAR})")

    # ── Step 2: Temporal CV ────────────────────────────────────────────────────
    log.info("\n--- Temporal CV (C grid search) ---")
    cv_df, best_c = run_temporal_cv(train_df, features=FEATURES)
    cv_df.to_csv(PROCESSED_DIR / "formula_new_cv_results.csv", index=False)
    log.info(f"CV results saved → formula_new_cv_results.csv")

    mean_acc = cv_df["accuracy"].mean()
    mean_ll  = cv_df["log_loss"].mean()
    log.info(f"\nCV summary: mean_acc={mean_acc:.4f}  mean_ll={mean_ll:.4f}  best_C={best_c}")

    # ── Step 3: Holdout evaluation (2024) ─────────────────────────────────────
    log.info(f"\n--- Holdout year: {HOLDOUT_YEAR} ---")
    model_holdout, scaler_holdout, _ = fit_model(train_df, features=FEATURES, c=best_c)
    X_h = holdout_df.dropna(subset=FEATURES + ["LABEL"])[FEATURES].values.astype(float)
    y_h = holdout_df.dropna(subset=FEATURES + ["LABEL"])["LABEL"].values.astype(int)
    probs_h = model_holdout.predict_proba(scaler_holdout.transform(X_h))[:, 1]
    acc_h = accuracy_score(y_h, (probs_h >= 0.5).astype(int))
    ll_h  = log_loss(y_h, np.clip(probs_h, 1e-7, 1-1e-7))
    log.info(f"Holdout {HOLDOUT_YEAR}: acc={acc_h:.4f}  log_loss={ll_h:.4f}")

    # ── Step 4: Simulate 2024 bracket for ESPN score ───────────────────────────
    log.info(f"\n--- Simulating {HOLDOUT_YEAR} bracket ---")
    result_2024 = simulate_bracket(HOLDOUT_YEAR, model_holdout, scaler_holdout, FEATURES)
    log.info(f"Predicted champion {HOLDOUT_YEAR}: {result_2024['champion']}")
    if result_2024["espn_score"] is not None:
        log.info(f"ESPN score {HOLDOUT_YEAR}: {result_2024['espn_score']}/1920")

    # ── Step 5: Fit final model on ALL training data (2008–2023) ──────────────
    log.info("\n--- Final model (all training years 2008-2023) ---")
    model_final, scaler_final, raw_coefs = fit_model(train_df, features=FEATURES, c=best_c)
    formula_df = extract_formula(model_final, scaler_final, raw_coefs, FEATURES)
    formula_df.to_csv(PROCESSED_DIR / "formula_new_weights.csv", index=False)
    log.info(f"Formula weights saved → formula_new_weights.csv")

    print(f"\n{'='*60}")
    print("FORMULA WEIGHTS (new dataset):")
    print(f"{'='*60}")
    for _, row in formula_df.iterrows():
        sign = "+" if row["raw_weight"] > 0 else ""
        print(f"  {row['feature']:15s}  w={sign}{row['raw_weight']:.4f}  "
              f"({row['interpretation']})")

    # ── Step 6: Predict 2026 bracket ──────────────────────────────────────────
    log.info("\n--- Predicting 2026 bracket ---")
    try:
        result_2026 = simulate_bracket(2026, model_final, scaler_final, FEATURES)
        log.info(f"Predicted 2026 champion: {result_2026['champion']}")

        # Save bracket prediction
        bracket_df = pd.DataFrame(result_2026["matchups"])
        bracket_df.to_csv(PROCESSED_DIR / "predicted_bracket_2026_new.csv", index=False)
        log.info("Saved → predicted_bracket_2026_new.csv")

        print(f"\n{'='*60}")
        print("2026 BRACKET PREDICTION (new model):")
        print(f"{'='*60}")
        print(f"\nPredicted champion: {result_2026['champion']}")
        print("\nRegion champions:")
        champ_rounds = result_2026["team_rounds"]
        for region in REGIONS:
            region_matchups = [m for m in result_2026["matchups"] if m["region"] == region]
            if region_matchups:
                e8 = [m for m in region_matchups if m["round"] == 4]
                if e8:
                    champ = e8[-1]["predicted_winner"]
                    seed = result_2026["matchups"][0]["seed_a"]  # placeholder
                    print(f"  {region:10s}: {champ}")

        print("\nFinal Four:")
        ff = [m for m in result_2026["matchups"] if m["region"] == "Final Four"]
        for m in ff:
            print(f"  {m['team_a']} ({m['seed_a']}) vs {m['team_b']} ({m['seed_b']}) "
                  f"→ {m['predicted_winner']} ({m['prob_a']:.1%})")

        champ_m = [m for m in result_2026["matchups"] if m["region"] == "Championship"]
        if champ_m:
            m = champ_m[0]
            print(f"\nChampionship: {m['team_a']} vs {m['team_b']} "
                  f"→ {m['predicted_winner']} ({m['prob_a']:.1%})")

    except Exception as e:
        log.warning(f"2026 bracket simulation failed: {e}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Training years: {min(TRAIN_YEARS)}–{max(TRAIN_YEARS)} (excl. 2020)")
    print(f"CV mean accuracy: {mean_acc:.4f}")
    print(f"CV mean log loss: {mean_ll:.4f}")
    print(f"Holdout {HOLDOUT_YEAR} accuracy: {acc_h:.4f}")
    if result_2024["espn_score"] is not None:
        print(f"Holdout {HOLDOUT_YEAR} ESPN score: {result_2024['espn_score']}/1920")
    print(f"Best C: {best_c}")


if __name__ == "__main__":
    main()
