"""
generate_2025_bracket.py — Generate the 2025 bracket prediction using the
formula model trained on 2013–2024 (all available data), with the REAL
Selection Sunday 2025 bracket regions hard-coded.

The 2025 bracket is a live prediction (not holdout), so we train on ALL
available years (2013–2024 excl. 2020) to get the best possible formula.

Run from project root:
    python generate_2025_bracket.py
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import PROCESSED_DIR
from src.models.formula_model import (
    load_matchup_data, load_features, fit_model, predict_prob, FEATURES
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Real 2025 NCAA Bracket (Selection Sunday, March 16 2025) ──────────────────
# Format: {region: {seed: team_name}}
# First Four already resolved using model win-probability (see below)
# Seeds 11 (6 teams → 4) and 16 (6 teams → 4) need First Four play-in resolution

REAL_2025_BRACKET: dict[str, dict[int, str]] = {
    "South": {
        1: "Auburn",
        2: "Michigan St.",
        3: "Iowa St.",
        4: "Texas A&M",
        5: "Michigan",
        6: "Mississippi",
        7: "Marquette",
        8: "Louisville",
        9: "Creighton",
        10: "New Mexico",
        11: "San Diego St.",   # First Four winner (vs VCU)
        12: "UC San Diego",
        13: "Yale",
        14: "Lipscomb",
        15: "Bryant",
        16: "Alabama St.",     # First Four winner (vs Saint Francis)
    },
    "East": {
        1: "Duke",
        2: "Alabama",
        3: "Wisconsin",
        4: "Arizona",
        5: "Oregon",
        6: "BYU",
        7: "Saint Mary's",
        8: "Mississippi St.",
        9: "Baylor",
        10: "Vanderbilt",
        11: "Drake",           # First Four winner (vs North Carolina)
        12: "Liberty",
        13: "Akron",
        14: "Montana",
        15: "Robert Morris",
        16: "American",        # First Four winner (vs Mount St. Mary's)
    },
    "West": {
        1: "Florida",
        2: "St. John's",
        3: "Texas Tech",
        4: "Maryland",
        5: "Memphis",
        6: "Missouri",
        7: "Kansas",
        8: "Connecticut",
        9: "Oklahoma",
        10: "Arkansas",
        11: "Texas",           # First Four winner (vs Xavier)
        12: "McNeese St.",
        13: "High Point",
        14: "Troy",
        15: "Nebraska Omaha",
        16: "Norfolk St.",     # First Four winner (vs SIU Edwardsville)
    },
    "Midwest": {
        1: "Houston",
        2: "Tennessee",
        3: "Kentucky",
        4: "Purdue",
        5: "Clemson",
        6: "Illinois",
        7: "UCLA",
        8: "Gonzaga",
        9: "Georgia",
        10: "Utah St.",
        11: "VCU",             # alt: North Carolina — will be resolved via model
        12: "Colorado St.",
        13: "Grand Canyon",
        14: "UNC Wilmington",
        15: "Wofford",
        16: "SIU Edwardsville", # alt — will be resolved via model
    },
}

# For First Four games we need to pick the right team.
# Let's resolve them via win probability rather than hard-coding.
FIRST_FOUR_GAMES = [
    # (team_a, team_b, region, seed)  — winner goes into that region at that seed
    ("VCU",            "San Diego St.",   "South",   11),
    ("Saint Francis",  "Alabama St.",     "South",   16),
    ("North Carolina", "Drake",           "East",    11),
    ("American",       "Mount St. Mary's","East",    16),
    ("Xavier",         "Texas",           "West",    11),
    ("Norfolk St.",    "SIU Edwardsville","West",    16),
]

SEED_PAIRINGS = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
REGIONS = ["South", "East", "West", "Midwest"]
ESPN_POINTS = {1:10, 2:20, 3:40, 4:80, 5:160, 6:320}


def resolve_first_four(
    bracket: dict[str, dict[int, str]],
    feat_lookup: dict[str, np.ndarray],
    model: LogisticRegression,
    scaler: StandardScaler,
    matchups_out: list[dict],
    seed_lookup: dict[str, int],
) -> dict[str, dict[int, str]]:
    """
    Simulate First Four play-in games and insert winners into the bracket.
    Also appends each First Four game to matchups_out.
    """
    for ta, tb, region, seed in FIRST_FOUR_GAMES:
        if ta in feat_lookup and tb in feat_lookup:
            p = predict_prob(feat_lookup[ta], feat_lookup[tb], model, scaler)
        else:
            p = 0.5
        winner = ta if p >= 0.5 else tb
        matchups_out.append({
            "year": 2025, "region": region, "round": 0,
            "round_name": "First Four",
            "team_a": ta, "team_b": tb,
            "seed_a": seed_lookup.get(ta, seed),
            "seed_b": seed_lookup.get(tb, seed),
            "prob_a": round(p, 4), "prob_b": round(1 - p, 4),
            "predicted_winner": winner,
            "actual_winner": None, "correct": None,
        })
        bracket[region][seed] = winner
        log.info(f"  First Four ({region} #{seed}): {winner} beats {'San Diego St.' if winner==ta else ta} ({p:.1%})")
    return bracket


def simulate_2025(
    model: LogisticRegression,
    scaler: StandardScaler,
    feat_lookup: dict[str, np.ndarray],
    seed_lookup: dict[str, int],
) -> list[dict]:
    """
    Simulate the full 2025 bracket using the real Selection Sunday seedings.

    Returns list of matchup dicts (round 0 = First Four, 1–6 = main bracket).
    """
    bracket = {r: dict(seeds) for r, seeds in REAL_2025_BRACKET.items()}
    matchups: list[dict] = []

    # Resolve First Four
    bracket = resolve_first_four(bracket, feat_lookup, model, scaler, matchups, seed_lookup)

    def game(ta: str, tb: str, rnd: int, region: str) -> dict:
        if ta in feat_lookup and tb in feat_lookup:
            p = predict_prob(feat_lookup[ta], feat_lookup[tb], model, scaler)
        else:
            p = 0.5
        winner = ta if p >= 0.5 else tb
        return {
            "year": 2025, "region": region, "round": rnd,
            "round_name": {1:"R64",2:"R32",3:"S16",4:"E8",5:"F4",6:"Champ"}.get(rnd, f"R{rnd}"),
            "team_a": ta, "team_b": tb,
            "seed_a": seed_lookup.get(ta), "seed_b": seed_lookup.get(tb),
            "prob_a": round(p, 4), "prob_b": round(1 - p, 4),
            "predicted_winner": winner,
            "actual_winner": None, "correct": None,
        }

    region_champs = {}
    for region in REGIONS:
        rt = bracket[region]
        current = []
        for high, low in SEED_PAIRINGS:
            ta = rt.get(high, f"Seed {high}")
            tb = rt.get(low,  f"Seed {low}")
            m = game(ta, tb, 1, region)
            matchups.append(m)
            current.append(m["predicted_winner"])
        for rnd in [2, 3, 4]:
            nxt = []
            for i in range(0, len(current), 2):
                if i + 1 < len(current):
                    m = game(current[i], current[i+1], rnd, region)
                    matchups.append(m)
                    nxt.append(m["predicted_winner"])
            current = nxt
        region_champs[region] = current[0] if current else None

    # Final Four: South vs East, West vs Midwest
    f4_winners = []
    for ra, rb in [(REGIONS[0], REGIONS[1]), (REGIONS[2], REGIONS[3])]:
        ta, tb = region_champs[ra], region_champs[rb]
        m = game(ta, tb, 5, "Final Four")
        matchups.append(m)
        f4_winners.append(m["predicted_winner"])

    # Championship
    m = game(f4_winners[0], f4_winners[1], 6, "Championship")
    matchups.append(m)

    return matchups


def main() -> None:
    log.info("Loading matchup data for 2013–2024 (excl. 2020)...")
    df = load_matchup_data()
    # For 2025 prediction, train on EVERYTHING available (2013–2024 excl. 2020)
    train_df = df[df["YEAR"].isin(list(range(2013, 2025))) & (df["YEAR"] != 2020)]
    log.info(f"  Training on {len(train_df)//2} games across {train_df['YEAR'].nunique()} seasons")

    log.info("Fitting formula model on 2013–2024...")
    model, scaler, _ = fit_model(train_df)

    log.info("Loading 2025 team features...")
    feats_df = load_features()
    yr2025 = feats_df[(feats_df["YEAR"] == 2025) & feats_df["SEED"].notna()].copy()
    yr2025 = yr2025.dropna(subset=FEATURES)

    feat_lookup = {row["TEAM"]: row[FEATURES].values for _, row in yr2025.iterrows()}
    seed_lookup  = dict(zip(yr2025["TEAM"], yr2025["SEED"].astype(int)))
    log.info(f"  {len(feat_lookup)} teams with full features")

    log.info("\nSimulating 2025 bracket...")
    matchups = simulate_2025(model, scaler, feat_lookup, seed_lookup)

    champion = next((m["predicted_winner"] for m in matchups if m["round"] == 6), None)
    log.info(f"\n  Predicted 2025 Champion: {champion}")

    # Print R64
    print("\n2025 BRACKET PREDICTIONS")
    print("=" * 60)
    for region in REGIONS:
        print(f"\n  {region.upper()}")
        for m in matchups:
            if m["region"] == region and m["round"] == 1:
                print(f"    #{m['seed_a']:2} {m['team_a']:22} vs #{m['seed_b']:2} {m['team_b']:22}  → {m['predicted_winner']} ({m['prob_a']:.0%})")
    print(f"\n  PREDICTED CHAMPION: {champion}")

    # Save
    out_path = PROCESSED_DIR / "predicted_bracket_2025.csv"
    pd.DataFrame(matchups).to_csv(out_path, index=False)
    log.info(f"\nSaved {len(matchups)} matchups to {out_path}")


if __name__ == "__main__":
    main()
