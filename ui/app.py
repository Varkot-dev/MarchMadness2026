"""
ui/app.py — Flask backend for the March Madness Bracket Visualizer.

Predictions come from formula_model_new.py (new 2008–2026 dataset):
  - Trained on 2008–2024 tournament data (strict temporal, no leakage)
  - Explicit formula: P(A beats B) = σ( SCORE(A) - SCORE(B) )
  - Features: WAB + TALENT + KADJ O (SHAP-selected, 16-year dataset)
  - Holdout predictions: 2022, 2023, 2024 (never seen during training)
  - 2026: live prediction using full 2008–2024 training data

Run from project root:
    python -m ui.app
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, render_template
import pandas as pd
import numpy as np


from config import PROCESSED_DIR, EXTERNAL_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)
app = Flask(__name__)


def _sanitize(obj):
    """Recursively replace float NaN/Inf with None so jsonify never emits invalid JSON.

    Flask 3.x removed json_encoder; sanitize data directly before passing to jsonify.
    """
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj

# ── Constants ─────────────────────────────────────────────────────────────────

ROUND_NAMES = {1:"Round of 64", 2:"Round of 32", 3:"Sweet 16",
               4:"Elite Eight", 5:"Final Four", 6:"Championship"}
ROUND_SHORT = {1:"R64", 2:"R32", 3:"S16", 4:"E8", 5:"F4", 6:"Champ"}
ESPN_POINTS = {1:10, 2:20, 3:40, 4:80, 5:160, 6:320}

REGIONS = ["South", "East", "West", "Midwest"]
SEED_PAIRINGS = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]

# DayNum → round inferred dynamically per year in load_actual_results_raw().
# Do not use a hardcoded table — DayNums shift across years (esp. 2021 bubble).
DAYNUM_TO_ROUND: dict[int, int] = {}  # populated dynamically

# Training split used in formula_model_new.py (new 16-year pipeline)
TRAIN_YEARS   = list(range(2008, 2025))
HOLDOUT_YEARS = [2022, 2023, 2024]
LIVE_YEAR     = 2026


# ── Data loading ──────────────────────────────────────────────────────────────

def load_formula_weights() -> dict:
    """
    Load formula weights. Prefers formula_new_weights.csv (new pipeline),
    falls back to formula_weights.csv (legacy).

    Returns:
        Dict with feature → raw_weight, plus metadata.
    """
    for name in ("formula_new_weights.csv", "formula_weights.csv"):
        path = PROCESSED_DIR / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        return {
            "weights": {
                row["feature"]: {
                    "raw_weight":  float(row["raw_weight"]),
                    "norm_weight": float(row["norm_weight"]),
                    "interpretation": row.get("interpretation", row["feature"]),
                }
                for _, row in df.iterrows()
            },
            "formula": "P(A beats B) = σ( SCORE(A) − SCORE(B) )",
            "score_formula": "SCORE(T) = Σ w_i × feature_i(T)",
        }
    return {}


def load_cv_results() -> list[dict]:
    """Load temporal CV results. Prefers formula_new_cv_results.csv."""
    for name in ("formula_new_cv_results.csv", "formula_cv_results.csv"):
        path = PROCESSED_DIR / name
        if path.exists():
            return pd.read_csv(path).to_dict("records")
    return []


def load_predicted_bracket(year: int) -> list[dict] | None:
    """
    Load pre-computed bracket predictions for a year.

    Tries predicted_bracket_{year}_new.csv first (new pipeline),
    then predicted_bracket_{year}.csv (legacy).

    Returns list of matchup dicts, or None if no file found.
    """
    for suffix in (f"_{year}_new.csv", f"_{year}.csv"):
        path = PROCESSED_DIR / f"predicted_bracket{suffix}"
        if path.exists():
            return pd.read_csv(path).to_dict("records")
    return None


def load_features_for_year(year: int) -> dict[str, dict]:
    """
    Load team features dict for a given year from features_new.csv.

    Returns dict of team_name → feature dict with new model features
    (WAB, TALENT, KADJ O, BADJ EM, SEED_DIVERGENCE).
    """
    # Prefer new dataset
    for fname in ("features_new.csv", "features_coaching.csv"):
        path = PROCESSED_DIR / fname
        if path.exists():
            break
    else:
        return {}

    df = pd.read_csv(path)
    yr = df[(df["YEAR"] == year) & df["SEED"].notna()].copy()

    def sf(v):
        try:
            f = float(v)
            return round(f, 3) if f == f else None
        except (TypeError, ValueError):
            return None

    result = {}
    for _, row in yr.iterrows():
        entry = {
            "team": row["TEAM"],
            "seed": int(row["SEED"]),
            "conf": str(row.get("CONF") or ""),
        }
        # New pipeline features
        for col in ("WAB", "TALENT", "KADJ O", "BADJ EM", "SEED_DIVERGENCE",
                    "KADJ EM", "KADJ D", "BADJ D", "EXP"):
            if col in row:
                entry[col.lower().replace(" ", "_")] = sf(row.get(col))
        # Legacy features (fallback)
        for col in ("TRUE_QUALITY_SCORE", "ADJOE", "ADJDE", "QMS", "COACH_PREMIUM"):
            if col in row:
                entry[col.lower()] = sf(row.get(col))
        result[row["TEAM"]] = entry
    return result


def load_actual_results_raw(year: int) -> dict[str, int]:
    """
    Load actual tournament results using Kaggle data + new team name mapping.

    Returns dict of team_name → furthest round reached (0–6).
    """
    rpath = EXTERNAL_DIR / "kaggle" / "MNCAATourneyDetailedResults.csv"
    tpath = EXTERNAL_DIR / "kaggle" / "MTeams.csv"
    if not rpath.exists() or not tpath.exists():
        return {}
    try:
        from src.features.new_matchup_builder import _build_kaggle_to_new_name
        feats_df = pd.read_csv(PROCESSED_DIR / "features_new.csv")
        teams_df = pd.read_csv(tpath)
        kaggle_to_cbb = _build_kaggle_to_new_name(feats_df, teams_df)
    except Exception as e:
        log.warning(f"Could not build team name map via new pipeline: {e}. Trying legacy.")
        try:
            from src.utils.team_names import build_kaggle_to_cbb_map
            feats = pd.read_csv(PROCESSED_DIR / "features_coaching.csv")
            teams_df = pd.read_csv(tpath)
            kaggle_to_cbb = build_kaggle_to_cbb_map(feats, teams_df)
        except Exception as e2:
            log.warning(f"Legacy map also failed: {e2}")
            return {}

    results_df = pd.read_csv(rpath)
    yr = results_df[results_df["Season"] == year]
    if yr.empty:
        return {}

    from src.utils.team_names import build_daynum_to_round
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


def get_available_years() -> list[int]:
    """Return years that have predicted bracket data."""
    years = set()
    for f in PROCESSED_DIR.glob("predicted_bracket_*.csv"):
        try:
            # handles both predicted_bracket_2024.csv and predicted_bracket_2024_new.csv
            stem = f.stem.replace("_new", "")
            years.add(int(stem.split("_")[-1]))
        except ValueError:
            pass
    return sorted(years)


def compute_espn_score(matchups: list[dict], actual: dict[str, int]) -> int | None:
    """Compute ESPN bracket score for a set of matchup predictions vs actual results."""
    if not actual:
        return None
    team_rounds: dict[str, int] = {}
    for m in matchups:
        w = m.get("predicted_winner")
        r = m.get("round", 0)
        if w:
            team_rounds[w] = max(team_rounds.get(w, 0), r)
        loser = m["team_b"] if m.get("predicted_winner") == m["team_a"] else m["team_a"]
        team_rounds[loser] = max(team_rounds.get(loser, 0), r - 1)
    score = 0
    for team, pred_rnd in team_rounds.items():
        for r in range(1, pred_rnd + 1):
            if actual.get(team, 0) >= r:
                score += ESPN_POINTS.get(r, 0)
    return score


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", years=get_available_years())


@app.route("/api/years")
def api_years():
    return jsonify({"years": get_available_years()})


@app.route("/api/bracket/<int:year>")
def api_bracket(year: int):
    """
    Return full bracket data for a given year.

    Returns 400 if the year has no available data.

    For 2022–2024: loads pre-computed holdout predictions.
    For 2026: loads live 2026 prediction from formula_model_new.py output.

    Response includes:
      - matchups: list of matchup dicts with predicted/actual winner + features
      - champion: predicted champion
      - actual_champion: actual champion (if available)
      - espn_score: ESPN score (if actual results available)
      - formula: the model weights
      - cv_results: temporal CV metrics
      - year_label: 'holdout' or 'live'
    """
    valid_years = get_available_years()
    if year not in valid_years:
        return jsonify({"error": f"Year {year} not available. Valid years: {valid_years}"}), 400

    matchups = load_predicted_bracket(year)
    if matchups is None:
        return jsonify({"error": f"No bracket data for {year}. Run formula_model_new.py first."}), 404

    is_holdout = year in HOLDOUT_YEARS

    # Attach full feature data to each matchup
    feats = load_features_for_year(year)
    for m in matchups:
        m["features_a"] = feats.get(m.get("team_a"), {})
        m["features_b"] = feats.get(m.get("team_b"), {})

    # Find predicted champion
    champion = None
    for m in matchups:
        if m.get("round") == 6:
            champion = m.get("predicted_winner")

    # Actual results (Kaggle data — not available for 2026)
    actual = load_actual_results_raw(year)
    actual_champion = None
    if actual:
        for t, r in actual.items():
            if r >= 6:
                actual_champion = t
                break

    # ESPN score + annotate matchups with actual_winner/correct
    espn_score = None
    if actual:
        espn_score = compute_espn_score(matchups, actual)
        if not is_holdout:
            for m in matchups:
                ta, tb, rnd = m.get("team_a"), m.get("team_b"), m.get("round", 0)
                if m.get("actual_winner") is None:
                    if actual.get(ta, -1) >= rnd:
                        m["actual_winner"] = ta
                    elif actual.get(tb, -1) >= rnd:
                        m["actual_winner"] = tb
                if m.get("correct") is None and m.get("actual_winner") is not None:
                    m["correct"] = m["predicted_winner"] == m["actual_winner"]

    # Round-by-round accuracy
    round_acc = {}
    for rnd in range(1, 7):
        rnd_matchups = [m for m in matchups if m.get("round") == rnd and m.get("correct") is not None]
        if rnd_matchups:
            n_correct = sum(1 for m in rnd_matchups if m["correct"])
            round_acc[ROUND_SHORT[rnd]] = {
                "correct": n_correct,
                "total": len(rnd_matchups),
                "pct": round(n_correct / len(rnd_matchups), 4),
            }

    formula_info = load_formula_weights()
    cv_results   = load_cv_results()

    if year == LIVE_YEAR:
        year_label = "live"
    elif is_holdout:
        year_label = "holdout"
    else:
        year_label = "training"

    return jsonify(_sanitize({
        "year":             year,
        "matchups":         matchups,
        "champion":         champion,
        "actual_champion":  actual_champion,
        "espn_score":       espn_score,
        "has_actual":       bool(actual),
        "is_holdout":       is_holdout,
        "year_label":       year_label,
        "round_accuracy":   round_acc,
        "formula":          formula_info,
        "cv_results":       cv_results,
        "train_years":      TRAIN_YEARS,
        "holdout_years":    HOLDOUT_YEARS,
    }))


@app.route("/api/formula")
def api_formula():
    """Return model formula weights and CV results."""
    return jsonify(_sanitize({
        "formula":       load_formula_weights(),
        "cv_results":    load_cv_results(),
        "train_years":   TRAIN_YEARS,
        "holdout_years": HOLDOUT_YEARS,
    }))


if __name__ == "__main__":
    print("=" * 52)
    print("  March Madness — Formula Model Visualizer")
    print("  http://localhost:5050")
    print("=" * 52)
    print(f"  Available years: {get_available_years()}")
    print(f"  Training years:  {TRAIN_YEARS[:3]} … {TRAIN_YEARS[-3:]}")
    print(f"  Holdout years:   {HOLDOUT_YEARS}")
    print(f"  Live year:       {LIVE_YEAR}")
    app.run(debug=True, port=5050, host="0.0.0.0")
