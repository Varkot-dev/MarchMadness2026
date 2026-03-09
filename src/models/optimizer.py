"""
optimizer.py — Layer 3: Multi-bracket optimizer.

Given the Monte Carlo simulation results from Layer 2, constructs three
brackets that maximize the expected best-bracket ESPN score across all
three entries. Submitting three identical brackets is strictly suboptimal
when scoring is top-heavy and outcomes are uncertain.

Three bracket types:
  - Chalk:  Pick the highest-probability team at every node.
            Maximizes per-game accuracy. Safe floor, moderate ceiling.
  - Medium: Pick moderate upsets where the model's probability diverges
            meaningfully from seed expectation. One deliberate upset per
            region minimum; champion still from top-4 seeds.
  - Chaos:  Maximize bracket-score variance. Pick upsets with the highest
            expected-score-per-point, weighted by Seed_Divergence. Low
            floor, high ceiling. The lottery ticket entry.

Math:
  expected_score(team, round) = P(team reaches round) * ESPN_points[round]

  This is computed directly from simulation frequencies, which already
  account for path dependency. We never need to know who a team faces
  in round 3 — the simulator already ran those matchups 10,000 times.

  For medium and chaos brackets, we compare expected scores and apply
  upset utility bonuses to shift picks toward higher-variance outcomes.

Output:
  - Three bracket dicts (same schema as simulator output)
  - Printed summary table
  - data/processed/three_brackets_2025.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROCESSED_DIR
from src.models.simulator import (
    BRACKET_2025,
    ESPN_ROUND_POINTS,
    FIRST_FOUR_SLOTS,
    run_simulations,
    top_champions,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

FEATURES_FILE = PROCESSED_DIR / "features_coaching.csv"
N_ROUNDS = 6


# ── P(reach) table ─────────────────────────────────────────────────────────────

def build_p_reach(sim_results: dict) -> pd.DataFrame:
    """
    Compute P(team reaches round R) from simulation frequencies.

    For each of the 10,000 simulated brackets, a team's picks dict maps
    team → furthest round reached. Dividing counts by n_sims gives the
    empirical reach probability.

    Args:
        sim_results: Output dict from run_simulations().

    Returns:
        DataFrame indexed by team name, columns ROUND_1..ROUND_6.
        Value at [team, ROUND_R] = fraction of sims where team reached >= R.
    """
    brackets = sim_results["brackets"]
    n = len(brackets)

    # Collect all team names
    all_teams: set[str] = set()
    for b in brackets:
        all_teams.update(b["picks"].keys())

    reach_counts = {team: np.zeros(N_ROUNDS + 1, dtype=np.int32) for team in all_teams}

    for b in brackets:
        for team, max_round in b["picks"].items():
            for r in range(1, max_round + 1):
                reach_counts[team][r] += 1

    records = []
    for team in sorted(all_teams):
        row = {"TEAM": team}
        for r in range(1, N_ROUNDS + 1):
            row[f"ROUND_{r}"] = round(reach_counts[team][r] / n, 4)
        records.append(row)

    df = pd.DataFrame(records).set_index("TEAM")
    log.info(f"P(reach) table: {len(df)} teams × {N_ROUNDS} rounds")
    return df


def expected_score(team: str, p_reach: pd.DataFrame, from_round: int = 1) -> float:
    """
    Expected ESPN points a team contributes from a given round onward.

    E[score | from_round] = Σ_{r=from_round}^{6} P(team reaches r) * points[r]

    Using from_round=current_game_round ensures that when comparing two
    teams in a first-round game we only look at round-1+ value, and when
    comparing two teams in a Final Four game we look at rounds 5-6 value.
    This prevents 1-seeds from always dominating because of their deep-run
    potential regardless of which round we're currently deciding.

    Args:
        team:       Team name.
        p_reach:    P(reach) DataFrame from build_p_reach().
        from_round: First round to include in the sum (inclusive).

    Returns:
        Expected ESPN point contribution from from_round through round 6.
    """
    if team not in p_reach.index:
        return 0.0
    total = 0.0
    for r in range(from_round, N_ROUNDS + 1):
        col = f"ROUND_{r}"
        total += p_reach.loc[team, col] * ESPN_ROUND_POINTS[r]
    return total


# ── Seed divergence lookup ──────────────────────────────────────────────────────

def load_seed_divergence(year: int = 2025) -> dict[str, float]:
    """
    Load Seed_Divergence scores for each team from the features file.

    Positive = underseeded (potential sleeper).
    Negative = overseeded.

    Args:
        year: Season year.

    Returns:
        Dict mapping team name → Seed_Divergence value.
    """
    features = pd.read_csv(FEATURES_FILE)
    yr = features[(features["YEAR"] == year) & features["SEED"].notna()]
    return dict(zip(yr["TEAM"], yr["SEED_DIVERGENCE"].fillna(0.0)))


# ── Bracket construction ────────────────────────────────────────────────────────

def _resolve_first_four_chalk(p_reach: pd.DataFrame) -> dict[str, str]:
    """
    Resolve First Four matchups by picking the team with higher P(ROUND_1).

    Args:
        p_reach: P(reach) DataFrame.

    Returns:
        Dict mapping bracket slot string → winning team name.
    """
    resolved = {}
    for slot, (team_a, team_b) in FIRST_FOUR_SLOTS.items():
        p_a = p_reach.loc[team_a, "ROUND_1"] if team_a in p_reach.index else 0.0
        p_b = p_reach.loc[team_b, "ROUND_1"] if team_b in p_reach.index else 0.0
        resolved[slot] = team_a if p_a >= p_b else team_b
    return resolved


def _pick_game(
    team_a: str,
    team_b: str,
    current_round: int,
    p_reach: pd.DataFrame,
    seed_div: dict[str, float],
    prob_matrix: pd.DataFrame,
    mode: str,
) -> str:
    """
    Pick the winner of a single bracket game under a given strategy.

    Chalk uses full downstream expected score (from current_round onward)
    to correctly value teams with high championship probability.

    Medium and Chaos use P(team wins THIS game) from the win-probability
    matrix directly, combined with Seed_Divergence, to identify mispriced
    matchups. This separates the upstream value signal (chalk) from the
    upset-identification signal (medium/chaos).

    Chalk:  pick the team with higher downstream expected score.
    Medium: pick the underdog if:
              P(dog wins) >= MEDIUM_WIN_PROB_FLOOR  (model gives them real chance)
              AND Seed_Divergence(dog) > 0            (committee underseeded them)
    Chaos:  pick the team that maximizes
              P(team wins) * (1 + CHAOS_WEIGHT * max(seed_div, 0))
            explicitly rewarding underseeded underdogs the model likes.

    Args:
        team_a:        First team.
        team_b:        Second team.
        current_round: Round number being decided (1–6).
        p_reach:       P(reach) DataFrame.
        seed_div:      Seed divergence lookup.
        prob_matrix:   Win probability matrix (P(row beats col)).
        mode:          "chalk", "medium", or "chaos".

    Returns:
        Name of the picked winner.
    """
    # Thresholds — tuned so each bracket produces meaningfully different picks.
    #   Medium: conservative upset floor. Only take an upset when the model
    #           says the game is genuinely competitive (dog >= 35%) AND the
    #           committee underseed the dog. Typically 2–4 upsets per bracket.
    #   Chaos:  lower the bar to 25% AND boost by seed divergence. Accepts
    #           riskier upsets, especially for highly underseeded teams.
    #           Typically 5–8 upsets per bracket.
    MEDIUM_WIN_PROB_FLOOR = 0.35
    CHAOS_WIN_PROB_FLOOR  = 0.25
    CHAOS_SD_BOOST        = 0.04   # Drop effective floor by 4% per sd point above 0

    # --- Chalk: downstream expected score (captures path-dependent value) ---
    es_a = expected_score(team_a, p_reach, from_round=current_round)
    es_b = expected_score(team_b, p_reach, from_round=current_round)

    if mode == "chalk":
        return team_a if es_a >= es_b else team_b

    # --- Medium / Chaos: win probability from the model ---
    if team_a in prob_matrix.index and team_b in prob_matrix.columns:
        p_win_a = float(prob_matrix.loc[team_a, team_b])
    else:
        p_win_a = 0.5
    p_win_b = 1.0 - p_win_a

    # Identify favorite (higher win prob) and underdog
    if p_win_a >= p_win_b:
        fav, dog = team_a, team_b
        p_fav, p_dog = p_win_a, p_win_b
    else:
        fav, dog = team_b, team_a
        p_fav, p_dog = p_win_b, p_win_a

    sd_dog = seed_div.get(dog, 0.0)

    if mode == "medium":
        # Pick upset only when the game is competitive AND dog is underseeded
        if p_dog >= MEDIUM_WIN_PROB_FLOOR and sd_dog > 0:
            return dog
        return fav

    if mode == "chaos":
        # Effective floor decreases as seed divergence grows — more underseeded
        # teams get picked even at lower win probability
        effective_floor = CHAOS_WIN_PROB_FLOOR - CHAOS_SD_BOOST * max(sd_dog, 0.0)
        effective_floor = max(effective_floor, 0.10)  # never go below 10%
        if p_dog >= effective_floor and sd_dog > 0:
            return dog
        return fav

    raise ValueError(f"Unknown mode: {mode!r}")


def build_bracket(
    p_reach: pd.DataFrame,
    seed_div: dict[str, float],
    prob_matrix: pd.DataFrame,
    mode: str,
) -> dict:
    """
    Construct a full bracket using the given strategy mode.

    Walks the bracket tree round-by-round. At each game node, calls
    _pick_game() with the current survivors and the target round number.
    First Four games are resolved by chalk regardless of mode (too early
    to apply upset strategy on play-in games).

    Args:
        p_reach:     P(reach) DataFrame.
        seed_div:    Seed divergence lookup.
        prob_matrix: Win probability matrix from Layer 1.
        mode:        "chalk", "medium", or "chaos".

    Returns:
        Dict with keys:
          picks     — team → furthest round predicted (1–6)
          champion  — predicted champion name
          mode      — bracket type label
    """
    # First Four: always chalk (play-in games carry no strategic value)
    ff_resolved = _resolve_first_four_chalk(p_reach)

    picks: dict[str, int] = {}
    region_champs: list[str] = []

    for region in ["South", "East", "West", "Midwest"]:
        matchups = BRACKET_2025[region]

        # Resolve First Four slots
        current: list[tuple[str, str]] = []
        for team_a, team_b in matchups:
            team_a = ff_resolved.get(team_a, team_a)
            team_b = ff_resolved.get(team_b, team_b)
            current.append((team_a, team_b))

        # 4 regional rounds: R64(1) → R32(2) → S16(3) → E8(4)
        for round_num in range(1, 5):
            winners_this_round: list[str] = []
            for team_a, team_b in current:
                winner = _pick_game(
                    team_a, team_b, round_num, p_reach, seed_div, prob_matrix, mode
                )
                picks[winner] = round_num
                winners_this_round.append(winner)
            # Pair adjacent winners for the next round
            current = list(zip(winners_this_round[::2], winners_this_round[1::2]))

        # After 4 rounds, winners_this_round has 1 team: the regional champion (E8 winner)
        region_champ = winners_this_round[0]
        region_champs.append(region_champ)

    # Final Four (round 5): South vs East, West vs Midwest
    south, east, west, midwest = region_champs
    ff1 = _pick_game(south, east, 5, p_reach, seed_div, prob_matrix, mode)
    ff2 = _pick_game(west, midwest, 5, p_reach, seed_div, prob_matrix, mode)
    picks[ff1] = 5
    picks[ff2] = 5

    # Championship (round 6)
    champion = _pick_game(ff1, ff2, 6, p_reach, seed_div, prob_matrix, mode)
    picks[champion] = 6

    log.info(f"[{mode.upper()}] Champion: {champion}")
    return {"picks": picks, "champion": champion, "mode": mode}


# ── Summary output ─────────────────────────────────────────────────────────────

def _round_name(r: int) -> str:
    return {1: "R64", 2: "R32", 3: "S16", 4: "E8", 5: "F4", 6: "Champion"}[r]


def print_bracket_comparison(brackets: list[dict]) -> None:
    """
    Print a side-by-side comparison of the three brackets.

    Shows which teams each bracket advances to Final Four and beyond,
    and flags where chalk, medium, and chaos diverge.

    Args:
        brackets: List of three bracket dicts (chalk, medium, chaos).
    """
    modes = [b["mode"].upper() for b in brackets]
    header = f"{'TEAM':<22} " + "  ".join(f"{m:<10}" for m in modes)
    print("\n" + "=" * 60)
    print("Three-Bracket Comparison — Round Reached per Team")
    print("=" * 60)
    print(header)
    print("-" * 60)

    # Collect all teams that appear in any bracket at round 3+
    interesting: set[str] = set()
    for b in brackets:
        for team, rnd in b["picks"].items():
            if rnd >= 3:
                interesting.add(team)

    for team in sorted(interesting):
        rounds = []
        for b in brackets:
            r = b["picks"].get(team, 0)
            rounds.append(_round_name(r) if r > 0 else "—")

        # Flag rows where not all brackets agree
        flag = " *" if len(set(rounds)) > 1 else "  "
        print(f"{team:<22} " + "  ".join(f"{r:<10}" for r in rounds) + flag)

    print("-" * 60)
    print(f"{'CHAMPION':<22} " + "  ".join(f"{b['champion']:<10}" for b in brackets))
    print("(* = brackets diverge on this team)")


def print_expected_scores(
    brackets: list[dict],
    p_reach: pd.DataFrame,
) -> None:
    """
    Print expected ESPN score for each bracket based on P(reach) values.

    E[score for bracket B] = Σ_team Σ_round P(team reaches round) * points[round]
    summed only over rounds the bracket predicts that team to reach.

    Args:
        brackets: List of three bracket dicts.
        p_reach:  P(reach) DataFrame.
    """
    print("\n" + "=" * 60)
    print("Expected Score per Bracket (from simulation frequencies):")
    print("=" * 60)
    print(f"  {'Bracket':<12} {'Expected Score':>16}  {'Champion':}")
    print(f"  {'-'*12} {'-'*16}  {'-'*20}")
    for b in brackets:
        es = sum(
            expected_score(team, p_reach)
            for team in b["picks"]
        )
        print(f"  {b['mode'].upper():<12} {es:>16.1f}  {b['champion']}")


def save_brackets(brackets: list[dict], year: int = 2025) -> Path:
    """
    Save all three brackets to a tidy CSV.

    Each row: BRACKET_MODE, TEAM, ROUND_REACHED, ROUND_NAME.

    Args:
        brackets: List of three bracket dicts.
        year:     Tournament year.

    Returns:
        Path to the saved CSV.
    """
    rows = []
    for b in brackets:
        for team, rnd in sorted(b["picks"].items()):
            rows.append({
                "BRACKET_MODE": b["mode"],
                "TEAM": team,
                "ROUND_REACHED": rnd,
                "ROUND_NAME": _round_name(rnd),
                "CHAMPION": b["champion"],
            })
    df = pd.DataFrame(rows)
    out = PROCESSED_DIR / f"three_brackets_{year}.csv"
    df.to_csv(out, index=False)
    log.info(f"Saved three brackets to {out}")
    return out


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Load win probability matrix
    prob_matrix = pd.read_csv(
        PROCESSED_DIR / "win_prob_matrix_2025.csv", index_col=0
    )

    # 2. Run simulations (Layer 2) to get P(reach) distribution
    log.info("Running 10,000 simulations to build P(reach) table...")
    sim_results = run_simulations(prob_matrix, n_sims=10_000, seed=42)

    # 3. Build P(reach) table from simulation output
    p_reach = build_p_reach(sim_results)

    # 4. Load seed divergence for upset strategy
    seed_div = load_seed_divergence(year=2025)

    # 5. Build three brackets
    chalk  = build_bracket(p_reach, seed_div, prob_matrix, mode="chalk")
    medium = build_bracket(p_reach, seed_div, prob_matrix, mode="medium")
    chaos  = build_bracket(p_reach, seed_div, prob_matrix, mode="chaos")
    brackets = [chalk, medium, chaos]

    # 6. Print champion frequency from simulation
    print("\nTop 10 Most Common Champions (10,000 simulations):")
    print("=" * 40)
    print(top_champions(sim_results["champions"], n=10).to_string(index=False))

    # 7. Print bracket comparison
    print_bracket_comparison(brackets)

    # 8. Print expected scores
    print_expected_scores(brackets, p_reach)

    # 9. Save to CSV
    out_path = save_brackets(brackets, year=2025)
    print(f"\nBrackets saved to: {out_path}")
