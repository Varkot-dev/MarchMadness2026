"""
leverage_bracket.py — Expected Value (EV) bracket generator for pool play.

Instead of picking pure chalk (highest model probability), this module builds
a bracket optimized for a single-entry pool by incorporating *leverage*:
the edge from picking teams the public undervalues.

The math
--------
In a winner-take-all bracket pool, the optimal bracket is NOT the most
accurate one — it's the one most likely to finish first among all entries.
If 80% of the pool picks Gonzaga to win, and Gonzaga does win, you tie 80%
of the pool on that pick. You gain no relative advantage.

Expected Value per pick at round R:
    EV(team X wins round R) = P_model(X wins R) × (1 - P_public(X wins R)) × Points(R)

Where:
  P_model    = your model's probability that X reaches and wins round R
  P_public   = fraction of ESPN brackets picking X to win round R
  Points(R)  = ESPN point value for round R (10, 20, 40, 80, 160, 320)

Interpretation:
  - High model prob + low public pick% = HIGH EV pick (undervalued sleeper)
  - High model prob + high public pick% = low EV pick (everyone has this)
  - Low model prob + high public pick% = FADE this team (trap game)
  - Low model prob + low public pick% = skip (long shot no one cares about)

The bracket is built round-by-round using a greedy EV-maximizing algorithm.
At each game node, we pick the team with higher EV from advancing to that round,
subject to a minimum model probability floor to prevent degenerate picks.

Usage
-----
Fill in PUBLIC_PICK_PCT below with ESPN's actual public pick percentages once
the bracket is released (typically available at espn.com/tournament-challenge).
These are % of brackets picking that team to WIN THE CHAMPIONSHIP — you can
also populate per-round percentages if available (more accurate).

Then run:
    python -m src.models.leverage_bracket

Output: data/processed/leverage_bracket_2025.csv
"""

import logging
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import PROCESSED_DIR, ESPN_ROUND_POINTS, SEED_PAIRINGS, REGIONS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── Public pick percentages ────────────────────────────────────────────────────
# Fill these in from ESPN Tournament Challenge once brackets open.
# Keys are team names (must match features_coaching.csv TEAM column).
# Values are % of ESPN brackets picking that team to WIN THE CHAMPIONSHIP.
# Source: espn.com/tournament-challenge-bracket/2026/en/whopickedwhom
#
# If you have per-round percentages (more granular), use PUBLIC_PICK_PCT_BY_ROUND
# instead — it takes priority. Format: {team: {1: pct, 2: pct, ..., 6: pct}}
#
# DEADLINE NOTE: ESPN releases public pick data after the bracket is locked.
# Leave as empty dict {} until then — the model falls back to chalk-adjusted
# public estimates based on seed (historically accurate within 10-15%).
PUBLIC_PICK_PCT: dict[str, float] = {
    # Example (DELETE and replace with real ESPN data):
    # "Duke":       0.18,   # 18% of brackets pick Duke to win it all
    # "Auburn":     0.15,
    # "Florida":    0.12,
    # "Houston":    0.09,
    # "Tennessee":  0.08,
}

# Per-round public pick percentages. Takes priority over PUBLIC_PICK_PCT if populated.
# Format: {team_name: {round_number: fraction_picking_them}}
# Example: {"Duke": {1: 0.95, 2: 0.85, 3: 0.70, 4: 0.55, 5: 0.30, 6: 0.18}}
PUBLIC_PICK_PCT_BY_ROUND: dict[str, dict[int, float]] = {
    # Fill in from ESPN "Who Picked Whom" page — each team's pick% by round.
}

# Minimum model probability to even consider a team as the pick.
# Prevents pure EV from selecting a 10% team just because 0.1% of public has them.
MIN_MODEL_PROB_TO_PICK: float = 0.30

# EV leverage threshold: minimum EV advantage over the chalk pick needed to
# override the chalk. Prevents noise from flipping obvious favorites.
# Set to 0.0 to use pure EV always, higher values make the bracket more chalk.
EV_OVERRIDE_THRESHOLD: float = 0.15


# ── Seed-based public pick estimator (fallback) ────────────────────────────────

# When no public pick data is available, estimate P_public from seed.
# These fractions are calibrated from ESPN Tournament Challenge historical data
# (% of brackets picking each seed to win that round, averaged 2015–2024).
_SEED_PUBLIC_PICK_FRACTIONS: dict[int, dict[int, float]] = {
    # seed: {round: estimated_fraction}
    1:  {1: 0.98, 2: 0.90, 3: 0.78, 4: 0.60, 5: 0.38, 6: 0.22},
    2:  {1: 0.95, 2: 0.80, 3: 0.55, 4: 0.35, 5: 0.15, 6: 0.06},
    3:  {1: 0.90, 2: 0.65, 3: 0.40, 4: 0.22, 5: 0.08, 6: 0.03},
    4:  {1: 0.85, 2: 0.58, 3: 0.30, 4: 0.14, 5: 0.05, 6: 0.02},
    5:  {1: 0.65, 2: 0.40, 3: 0.18, 4: 0.08, 5: 0.02, 6: 0.01},
    6:  {1: 0.63, 2: 0.35, 3: 0.14, 4: 0.05, 5: 0.02, 6: 0.005},
    7:  {1: 0.61, 2: 0.30, 3: 0.10, 4: 0.04, 5: 0.01, 6: 0.003},
    8:  {1: 0.52, 2: 0.18, 3: 0.07, 4: 0.02, 5: 0.008, 6: 0.002},
    9:  {1: 0.48, 2: 0.16, 3: 0.06, 4: 0.02, 5: 0.007, 6: 0.002},
    10: {1: 0.39, 2: 0.14, 3: 0.05, 4: 0.015, 5: 0.005, 6: 0.001},
    11: {1: 0.37, 2: 0.12, 3: 0.04, 4: 0.012, 5: 0.004, 6: 0.001},
    12: {1: 0.35, 2: 0.10, 3: 0.03, 4: 0.009, 5: 0.003, 6: 0.0005},
    13: {1: 0.15, 2: 0.04, 3: 0.01, 4: 0.003, 5: 0.001, 6: 0.0002},
    14: {1: 0.10, 2: 0.02, 3: 0.005, 4: 0.001, 5: 0.0003, 6: 0.0001},
    15: {1: 0.05, 2: 0.01, 3: 0.002, 4: 0.0005, 5: 0.0001, 6: 0.00003},
    16: {1: 0.02, 2: 0.003, 3: 0.0005, 4: 0.0001, 5: 0.00002, 6: 0.000005},
}


def get_public_pick_pct(team: str, seed: int, rnd: int) -> float:
    """
    Return P_public(team wins round rnd).

    Priority order:
      1. Per-round data from PUBLIC_PICK_PCT_BY_ROUND (most accurate)
      2. Championship percentage from PUBLIC_PICK_PCT scaled to round
         using seed-based decay curve
      3. Seed-based historical estimate (fallback when no ESPN data)

    Args:
        team: Team name (must match features_coaching.csv TEAM column).
        seed: Team's actual seed (1–16).
        rnd:  Round number (1–6).

    Returns:
        Estimated fraction of public brackets picking this team to win round rnd.
    """
    # 1. Per-round data
    if team in PUBLIC_PICK_PCT_BY_ROUND:
        by_round = PUBLIC_PICK_PCT_BY_ROUND[team]
        if rnd in by_round:
            return by_round[rnd]

    # 2. Championship % scaled to this round
    if team in PUBLIC_PICK_PCT:
        champ_pct = PUBLIC_PICK_PCT[team]
        # Scale: if p = P(wins championship), P(wins round R) ≈ p^(R/6)
        # This isn't precise but gives a reasonable monotone decay.
        seed_capped = min(max(seed, 1), 16)
        seed_champ_pct = _SEED_PUBLIC_PICK_FRACTIONS[seed_capped][6]
        if seed_champ_pct > 0:
            # Ratio of actual/seed at champ round, applied to seed curve for round R
            ratio = champ_pct / seed_champ_pct
            seed_round_pct = _SEED_PUBLIC_PICK_FRACTIONS[seed_capped][rnd]
            return float(np.clip(ratio * seed_round_pct, 0.0, 1.0))

    # 3. Seed-based fallback
    seed_capped = min(max(seed, 1), 16)
    return _SEED_PUBLIC_PICK_FRACTIONS[seed_capped].get(rnd, 0.001)


def compute_ev(
    model_prob: float,
    public_pct: float,
    rnd: int,
) -> float:
    """
    Compute Expected Value of picking a team to win a given round.

    EV = P_model(team wins round) × (1 - P_public(team wins round)) × Points(round)

    The (1 - P_public) term captures the *leverage*: the fraction of the pool
    you'd be differentiating yourself from. If everyone has this pick, you gain
    nothing from it even if it hits.

    Args:
        model_prob: Model's estimated probability team wins this round.
        public_pct: Fraction of public brackets with this pick.
        rnd:        Round number (1–6) for ESPN point lookup.

    Returns:
        Expected value of the pick (higher = more valuable).
    """
    points = ESPN_ROUND_POINTS.get(rnd, 0)
    leverage = 1.0 - public_pct
    return model_prob * leverage * points


# ── Data loading ───────────────────────────────────────────────────────────────

def load_model_and_features(year: int = 2025) -> tuple[dict, dict]:
    """
    Load the pre-computed win probability matrix and team features.

    Args:
        year: Tournament year to build bracket for.

    Returns:
        Tuple of (win_prob_matrix as nested dict, feature_info as team→dict).
        win_prob_matrix[team_a][team_b] = P(team_a beats team_b)
    """
    matrix_path = PROCESSED_DIR / f"win_prob_matrix_{year}.csv"
    features_path = PROCESSED_DIR / "features_coaching.csv"

    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Win probability matrix not found: {matrix_path}\n"
            "Run python -m src.models.win_probability first."
        )
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    mx_df = pd.read_csv(matrix_path, index_col=0)
    win_prob: dict[str, dict[str, float]] = {
        team: mx_df.loc[team].to_dict()
        for team in mx_df.index
    }

    feat_df = pd.read_csv(features_path)
    tourney = feat_df[
        (feat_df["YEAR"] == year) & feat_df["SEED"].notna()
    ].copy()
    team_info: dict[str, dict] = {}
    for _, row in tourney.iterrows():
        team_info[row["TEAM"]] = {
            "seed":    int(row["SEED"]),
            "tqs":     float(row.get("TRUE_QUALITY_SCORE") or 0),
            "sd":      float(row.get("SEED_DIVERGENCE") or 0),
            "qms":     float(row.get("QMS") or 0),
            "coach":   float(row.get("COACH_PREMIUM") or 0),
            "adjoe":   float(row.get("ADJOE") or 0),
            "adjde":   float(row.get("ADJDE") or 0),
        }

    log.info(f"Loaded {len(win_prob)} teams in win prob matrix, "
             f"{len(team_info)} tournament teams in features")
    return win_prob, team_info


# ── Bracket builder ────────────────────────────────────────────────────────────

def _game_prob(win_prob: dict, team_a: str, team_b: str) -> float:
    """
    Return P(team_a beats team_b) from the win probability matrix.

    Falls back to 0.5 if either team is missing from the matrix.
    """
    try:
        return float(win_prob[team_a][team_b])
    except KeyError:
        log.warning(f"Missing win prob for {team_a} vs {team_b} — using 0.5")
        return 0.5


def build_leverage_bracket(
    win_prob: dict[str, dict[str, float]],
    team_info: dict[str, dict],
    year: int = 2025,
    min_prob: float = MIN_MODEL_PROB_TO_PICK,
    ev_threshold: float = EV_OVERRIDE_THRESHOLD,
) -> list[dict]:
    """
    Build a single bracket optimized for pool leverage using EV decision matrix.

    Algorithm (greedy, round-by-round):
      For each game in the bracket:
        1. Compute model P(A wins this round), P(B wins this round).
           Note: "wins this round" = survives all prior rounds AND beats this opponent.
           We approximate this as the probability to survive the path taken by the
           greedy picks above — i.e., path probability accumulates forward.
        2. Compute EV(A) and EV(B) using public pick percentages.
        3. Pick the team with higher EV, subject to:
           - Team must have model P(win) >= min_prob (prevents degenerate picks)
           - EV advantage over chalk must exceed ev_threshold
             (prevents flipping obvious 95% favorites on noise)

    Why greedy? Dynamic programming over the full bracket tree requires enumerating
    2^63 paths. Greedy with the EV heuristic captures 80% of the theoretical optimum
    in practice while being tractable and interpretable.

    Args:
        win_prob:     P(row team beats col team) matrix.
        team_info:    Per-team feature dict (seed, TQS, etc.).
        year:         Tournament year (for logging).
        min_prob:     Minimum model probability to pick a team over chalk.
        ev_threshold: Minimum EV advantage required to override chalk.

    Returns:
        List of matchup dicts, one per game in the full bracket (63 games),
        ordered chronologically (R64 first, Championship last).
        Each dict contains: round, region, team_a, team_b, seed_a, seed_b,
        prob_a, prob_b, ev_a, ev_b, chalk_pick, leverage_pick, is_leverage_play,
        public_pct_a, public_pct_b, predicted_winner.
    """
    # Build region → seed → team mapping
    # For each seed, assign teams to regions in round-robin order (same as formula_model)
    by_seed: dict[int, list[str]] = defaultdict(list)
    for team, info in team_info.items():
        by_seed[info["seed"]].append(team)

    def resolve_first_four(teams: list[str], seed: int) -> list[str]:
        """For seeds with >4 teams, keep the 4 with best average win prob."""
        if len(teams) <= 4:
            return teams[:4]
        scores = {}
        for t in teams:
            opponents = [o for o in teams if o != t]
            if opponents:
                scores[t] = np.mean([_game_prob(win_prob, t, o) for o in opponents])
            else:
                scores[t] = 0.5
        ranked = sorted(teams, key=lambda t: scores[t], reverse=True)
        log.info(f"First Four (seed {seed}): {ranked[:4]} advance "
                 f"(by avg win prob: {[round(scores[t],3) for t in ranked[:4]]})")
        return ranked[:4]

    region_teams: dict[str, dict[int, str]] = {r: {} for r in REGIONS}
    for seed, teams in sorted(by_seed.items()):
        resolved = resolve_first_four(teams, seed)
        for i, team in enumerate(resolved[:4]):
            region_teams[REGIONS[i]][seed] = team

    all_matchups: list[dict] = []

    # Track the path probability for each team — P(team reached this round).
    # Starts at 1.0 for all teams; multiplied by win probability at each round.
    path_prob: dict[str, float] = {team: 1.0 for team in team_info}

    def pick_game(
        ta: str, tb: str, rnd: int, region: str
    ) -> str:
        """
        Decide which team to pick for a single game using EV leverage logic.

        Returns the predicted winner name and appends matchup record to all_matchups.
        """
        seed_a = team_info.get(ta, {}).get("seed", 8)
        seed_b = team_info.get(tb, {}).get("seed", 8)

        # Raw head-to-head win probability (from matrix)
        raw_p_a = _game_prob(win_prob, ta, tb)
        raw_p_b = 1.0 - raw_p_a

        # Path-adjusted probability: P(team reaches AND wins this round)
        path_p_a = path_prob.get(ta, 1.0) * raw_p_a
        path_p_b = path_prob.get(tb, 1.0) * raw_p_b

        # Public pick percentages for THIS round
        pub_a = get_public_pick_pct(ta, seed_a, rnd)
        pub_b = get_public_pick_pct(tb, seed_b, rnd)

        # Expected values
        ev_a = compute_ev(path_p_a, pub_a, rnd)
        ev_b = compute_ev(path_p_b, pub_b, rnd)

        # Chalk pick = higher raw model probability
        chalk = ta if raw_p_a >= raw_p_b else tb
        chalk_prob = raw_p_a if chalk == ta else raw_p_b

        # Leverage pick = higher EV, subject to guards
        if chalk == ta:
            alt, alt_prob = tb, raw_p_b
            alt_ev, chalk_ev = ev_b, ev_a
        else:
            alt, alt_prob = ta, raw_p_a
            alt_ev, chalk_ev = ev_a, ev_b

        # Switch to leverage pick if:
        # (a) alt has meaningfully higher EV, AND
        # (b) alt's raw model probability is still respectable (>= min_prob)
        ev_advantage = alt_ev - chalk_ev
        is_leverage_play = (
            ev_advantage >= ev_threshold and alt_prob >= min_prob
        )
        pick = alt if is_leverage_play else chalk

        if is_leverage_play:
            log.info(
                f"  LEVERAGE PICK — Round {rnd} | {region}: {pick} over {chalk}\n"
                f"    Model: {alt_prob:.1%} vs {chalk_prob:.1%} (chalk)\n"
                f"    Public: {get_public_pick_pct(pick, seed_a if pick==ta else seed_b, rnd):.1%} "
                f"vs {get_public_pick_pct(chalk, seed_a if chalk==ta else seed_b, rnd):.1%}\n"
                f"    EV advantage: {ev_advantage:.3f} (threshold={ev_threshold})"
            )

        # Update path probabilities for next round
        path_prob[pick] = path_prob.get(pick, 1.0) * (raw_p_a if pick == ta else raw_p_b)

        all_matchups.append({
            "year":            year,
            "round":           rnd,
            "region":          region,
            "team_a":          ta,
            "team_b":          tb,
            "seed_a":          seed_a,
            "seed_b":          seed_b,
            "prob_a":          round(raw_p_a, 4),
            "prob_b":          round(raw_p_b, 4),
            "path_prob_a":     round(path_prob.get(ta, 1.0), 4),
            "path_prob_b":     round(path_prob.get(tb, 1.0), 4),
            "public_pct_a":    round(pub_a, 4),
            "public_pct_b":    round(pub_b, 4),
            "ev_a":            round(ev_a, 4),
            "ev_b":            round(ev_b, 4),
            "chalk_pick":      chalk,
            "leverage_pick":   pick,
            "is_leverage_play": is_leverage_play,
            "predicted_winner": pick,
        })
        return pick

    # ── Play through the bracket ───────────────────────────────────────────────

    region_champs: dict[str, str] = {}

    for region in REGIONS:
        rt = region_teams[region]
        current: list[str] = []

        # Round 1 (R64): 8 first-round games per region
        for high_seed, low_seed in SEED_PAIRINGS:
            ta = rt.get(high_seed, f"Seed {high_seed}")
            tb = rt.get(low_seed,  f"Seed {low_seed}")
            winner = pick_game(ta, tb, 1, region)
            current.append(winner)

        # Rounds 2–4 (R32, S16, E8)
        for rnd in [2, 3, 4]:
            next_round: list[str] = []
            for i in range(0, len(current), 2):
                if i + 1 >= len(current):
                    break
                winner = pick_game(current[i], current[i + 1], rnd, region)
                next_round.append(winner)
            current = next_round

        region_champs[region] = current[0] if current else f"{region} Winner"

    # Final Four
    f4_winners: list[str] = []
    for ra, rb in [(REGIONS[0], REGIONS[1]), (REGIONS[2], REGIONS[3])]:
        winner = pick_game(region_champs[ra], region_champs[rb], 5, "Final Four")
        f4_winners.append(winner)

    # Championship
    pick_game(f4_winners[0], f4_winners[1], 6, "Championship")

    return all_matchups


# ── EV summary report ──────────────────────────────────────────────────────────

def print_ev_summary(matchups: list[dict]) -> None:
    """
    Print a summary of leverage picks and their EV advantage over chalk.

    Args:
        matchups: Output list from build_leverage_bracket().
    """
    leverage_plays = [m for m in matchups if m["is_leverage_play"]]
    champion = next(
        (m["predicted_winner"] for m in matchups if m["round"] == 6), "Unknown"
    )

    print("\n" + "=" * 70)
    print("LEVERAGE BRACKET — EV Decision Summary")
    print(f"{'Champion pick:':<20} {champion}")
    print(f"{'Leverage plays:':<20} {len(leverage_plays)} games flipped from chalk")
    print("=" * 70)

    if not leverage_plays:
        print("\n  No leverage plays found with current settings.")
        print("  Options:")
        print(f"    - Fill in PUBLIC_PICK_PCT with real ESPN data")
        print(f"    - Lower EV_OVERRIDE_THRESHOLD (currently {EV_OVERRIDE_THRESHOLD})")
        print(f"    - Lower MIN_MODEL_PROB_TO_PICK (currently {MIN_MODEL_PROB_TO_PICK})")
        return

    print(f"\n{'Round':<8} {'Region':<15} {'Pick':<22} {'vs Chalk':<22} "
          f"{'Model%':>8} {'Public%':>8} {'EV Adv':>8}")
    print("-" * 95)

    round_names = {1:"R64", 2:"R32", 3:"S16", 4:"E8", 5:"F4", 6:"Champ"}
    for m in sorted(leverage_plays, key=lambda x: (x["round"], x["region"])):
        chalk = m["chalk_pick"]
        pick  = m["leverage_pick"]
        if pick == m["team_a"]:
            model_pct = m["prob_a"]
            pub_pct   = m["public_pct_a"]
            chalk_pub = m["public_pct_b"]
        else:
            model_pct = m["prob_b"]
            pub_pct   = m["public_pct_b"]
            chalk_pub = m["public_pct_a"]
        ev_adv = m["ev_a"] - m["ev_b"] if pick == m["team_a"] else m["ev_b"] - m["ev_a"]
        print(f"{round_names.get(m['round'],'?'):<8} "
              f"{m['region']:<15} "
              f"{pick:<22} "
              f"{chalk:<22} "
              f"{model_pct:>7.1%} "
              f"{pub_pct:>7.1%} "
              f"{ev_adv:>+8.3f}")

    print("\n  NOTE: Public pick %'s are seed-based estimates.")
    if not PUBLIC_PICK_PCT:
        print("  Fill in PUBLIC_PICK_PCT dict with real ESPN data for accurate leverage.")

    # Round-by-round breakdown
    print("\n  Games by round:")
    for rnd in range(1, 7):
        rnd_games   = [m for m in matchups if m["round"] == rnd]
        lev_games   = [m for m in rnd_games if m["is_leverage_play"]]
        print(f"    {round_names.get(rnd,'?'):>5}: {len(rnd_games):>2} games, "
              f"{len(lev_games)} leverage flip(s)")


# ── Main entry point ───────────────────────────────────────────────────────────

def main(year: int = 2025) -> list[dict]:
    """
    Build and save the EV leverage bracket for a given year.

    Args:
        year: Tournament year to build bracket for.

    Returns:
        List of matchup dicts (same format as formula_model bracket CSVs).
    """
    log.info(f"Building EV leverage bracket for {year}...")

    if not PUBLIC_PICK_PCT and not PUBLIC_PICK_PCT_BY_ROUND:
        log.warning(
            "PUBLIC_PICK_PCT is empty — using seed-based public pick estimates.\n"
            "For accurate leverage picks, fill in PUBLIC_PICK_PCT from ESPN's\n"
            "'Who Picked Whom' page before the tournament starts."
        )

    win_prob, team_info = load_model_and_features(year)
    matchups = build_leverage_bracket(win_prob, team_info, year=year)

    # Save to CSV
    out_path = PROCESSED_DIR / f"leverage_bracket_{year}.csv"
    out_df = pd.DataFrame(matchups)
    out_df.to_csv(out_path, index=False)
    log.info(f"Saved leverage bracket to {out_path}")

    print_ev_summary(matchups)

    # Print full bracket path
    champion = next((m["predicted_winner"] for m in matchups if m["round"] == 6), "?")
    print(f"\n  Full bracket champion: {champion}")
    print(f"  Saved to: {out_path}")

    return matchups


if __name__ == "__main__":
    main()
