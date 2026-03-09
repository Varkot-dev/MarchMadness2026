"""
simulator.py — Layer 2: Monte Carlo tournament simulator.

Runs N simulations of the full 68-team NCAA bracket using win probabilities
from the Layer 1 model. Each game is sampled probabilistically — not always
picking the favorite.

Bracket structure:
  - First Four (4 games): 2 games among seed-11 teams, 2 among seed-16 teams
  - Main bracket: 64 teams across 4 regions (South, East, West, Midwest)
  - Standard 1v16, 2v15, ... 8v9 first-round seeding

ESPN scoring used per the competition spec:
  Round of 64:  10 pts
  Round of 32:  20 pts
  Sweet 16:     40 pts
  Elite 8:      80 pts
  Final Four:  160 pts
  Champion:    320 pts

Output:
  - Best bracket (maximizes 90th percentile score across simulations)
  - Top champion picks with frequency
  - Score distribution plot
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

ESPN_ROUND_POINTS = {
    1: 10,   # Round of 64
    2: 20,   # Round of 32
    3: 40,   # Sweet 16
    4: 80,   # Elite 8
    5: 160,  # Final Four
    6: 320,  # Championship
}

# 2025 NCAA Tournament bracket — First Four matchups.
# Each tuple: (team_a, team_b) — winner advances to main bracket slot.
FIRST_FOUR_2025 = [
    # Seed-11 play-ins
    ("Drake",         "North Carolina"),   # → South region 11-slot
    ("San Diego St.", "Texas"),            # → East region 11-slot (or West — adjust if needed)
    # Seed-16 play-ins
    ("American",      "Mount St. Mary's"), # → South region 16-slot
    ("Alabama St.",   "Saint Francis"),    # → Midwest region 16-slot
]

# 2025 Main bracket: 4 regions, each with 16 seeds.
# Format: list of 8 first-round matchups per region (seed order: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15)
# First Four winners are marked as "FF:team_a/team_b" — resolved before simulation.
BRACKET_2025 = {
    "South": [
        ("Auburn",       "Alabama St./Saint Francis"),   # 1 vs 16 (FF winner)
        ("Louisville",   "Creighton"),                   # 8 vs 9  (note: Creighton is 9, Louisville is 8)
        ("Michigan",     "UC San Diego"),                # 5 vs 12
        ("Texas A&M",    "Yale"),                        # 4 vs 13
        ("Mississippi",  "Drake/North Carolina"),        # 6 vs 11 (FF winner)
        ("Iowa St.",     "Lipscomb"),                    # 3 vs 14
        ("Marquette",    "New Mexico"),                  # 7 vs 10
        ("Michigan St.", "Bryant"),                      # 2 vs 15
    ],
    "East": [
        ("Duke",         "American/Mount St. Mary's"),   # 1 vs 16 (FF winner)
        ("Mississippi St.", "Baylor"),                   # 8 vs 9
        ("Oregon",       "Liberty"),                     # 5 vs 12
        ("Arizona",      "Akron"),                       # 4 vs 13
        ("Illinois",     "San Diego St./Texas"),         # 6 vs 11 (FF winner)
        ("Wisconsin",    "Montana"),                     # 3 vs 14
        ("Saint Mary's", "Vanderbilt"),                  # 7 vs 10
        ("Alabama",      "Robert Morris"),               # 2 vs 15
    ],
    "West": [
        ("Florida",      "Norfolk St."),                 # 1 vs 16
        ("Connecticut",  "Oklahoma"),                    # 8 vs 9
        ("Memphis",      "Colorado St."),                # 5 vs 12
        ("Maryland",     "Grand Canyon"),                # 4 vs 13
        ("Missouri",     "Drake/North Carolina"),        # placeholder — see note
        ("Kentucky",     "Troy"),                        # 3 vs 14
        ("UCLA",         "Utah St."),                    # 7 vs 10
        ("Tennessee",    "Wofford"),                     # 2 vs 15
    ],
    "Midwest": [
        ("Houston",      "SIU Edwardsville"),            # 1 vs 16
        ("Gonzaga",      "Georgia"),                     # 8 vs 9
        ("Clemson",      "McNeese St."),                 # 5 vs 12
        ("Purdue",       "High Point"),                  # 4 vs 13
        ("BYU",          "VCU"),                         # 6 vs 11
        ("Texas Tech",   "UNC Wilmington"),              # 3 vs 14
        ("Kansas",       "Arkansas"),                    # 7 vs 10
        ("St. John's",   "Nebraska Omaha"),              # 2 vs 15
    ],
}

# Resolved First Four slots: maps "placeholder/string" → (team_a, team_b)
FIRST_FOUR_SLOTS = {
    "Alabama St./Saint Francis":    ("Alabama St.",   "Saint Francis"),
    "American/Mount St. Mary's":    ("American",      "Mount St. Mary's"),
    "Drake/North Carolina":         ("Drake",          "North Carolina"),
    "San Diego St./Texas":          ("San Diego St.", "Texas"),
}


# ── Core simulation ────────────────────────────────────────────────────────────

def sim_game(team_a: str, team_b: str, prob_matrix: pd.DataFrame, rng: np.random.Generator) -> str:
    """
    Simulate a single game using win probability matrix.

    Args:
        team_a:      First team name.
        team_b:      Second team name.
        prob_matrix: Square DataFrame of P(row beats col).
        rng:         NumPy random generator.

    Returns:
        Name of the winning team.
    """
    p = prob_matrix.loc[team_a, team_b]
    return team_a if rng.random() < p else team_b


def resolve_first_four(prob_matrix: pd.DataFrame, rng: np.random.Generator) -> dict[str, str]:
    """
    Simulate the four First Four games and return slot → winner mapping.

    Args:
        prob_matrix: Win probability matrix.
        rng:         NumPy random generator.

    Returns:
        Dict mapping bracket slot string → winning team name.
    """
    return {
        slot: sim_game(teams[0], teams[1], prob_matrix, rng)
        for slot, teams in FIRST_FOUR_SLOTS.items()
    }


def sim_region(matchups: list[tuple], prob_matrix: pd.DataFrame, rng: np.random.Generator, ff_winners: dict[str, str]) -> tuple[list[list[str]], str]:
    """
    Simulate one region (4 rounds: R64 → R32 → S16 → E8).

    Args:
        matchups:   8 first-round matchups for this region.
        prob_matrix: Win probability matrix.
        rng:         NumPy random generator.
        ff_winners:  First Four slot → winner mapping.

    Returns:
        Tuple of (round_winners_per_round, regional_champion).
        round_winners_per_round[0] = 8 R64 winners, [1] = 4 R32 winners, etc.
    """
    # Resolve any First Four slots
    current = []
    for team_a, team_b in matchups:
        team_a = ff_winners.get(team_a, team_a)
        team_b = ff_winners.get(team_b, team_b)
        current.append((team_a, team_b))

    all_rounds = []
    teams = current
    while len(teams) > 1:
        winners = [sim_game(a, b, prob_matrix, rng) for a, b in teams]
        all_rounds.append(winners)
        teams = list(zip(winners[::2], winners[1::2]))

    # The regional champion (E8 winner) is the last single team — record it explicitly
    champion = all_rounds[-1][0]
    all_rounds.append([champion])
    return all_rounds, champion


def sim_final_four(region_champs: list[str], prob_matrix: pd.DataFrame, rng: np.random.Generator) -> tuple[str, str, str]:
    """
    Simulate Final Four and Championship (rounds 5 and 6).

    Bracket pairing: South vs East, West vs Midwest.

    Args:
        region_champs: [South, East, West, Midwest] champions.
        prob_matrix:   Win probability matrix.
        rng:           NumPy random generator.

    Returns:
        Tuple of (ff_winner_1, ff_winner_2, champion).
    """
    south, east, west, midwest = region_champs
    ff1 = sim_game(south, east, prob_matrix, rng)
    ff2 = sim_game(west, midwest, prob_matrix, rng)
    champion = sim_game(ff1, ff2, prob_matrix, rng)
    return ff1, ff2, champion


def sim_bracket(prob_matrix: pd.DataFrame, rng: np.random.Generator) -> dict:
    """
    Simulate one full 68-team tournament bracket.

    Args:
        prob_matrix: Win probability matrix.
        rng:         NumPy random generator.

    Returns:
        Dict with keys: picks (team → furthest round reached), champion.
        picks maps team_name → round_number (1–6), 0 = first round loss.
    """
    ff_winners = resolve_first_four(prob_matrix, rng)

    region_results = {}
    region_champs = []
    regions = ["South", "East", "West", "Midwest"]

    for region in regions:
        rounds, champ = sim_region(BRACKET_2025[region], prob_matrix, rng, ff_winners)
        region_results[region] = rounds
        region_champs.append(champ)

    ff1, ff2, champion = sim_final_four(region_champs, prob_matrix, rng)

    # Build picks: team → round reached (1=R64 winner, ..., 6=champion)
    picks: dict[str, int] = {}
    for region_idx, region in enumerate(regions):
        rounds = region_results[region]
        for round_idx, winners in enumerate(rounds):
            round_num = round_idx + 1
            for team in winners:
                picks[team] = round_num

    # Final Four (round 5) and Champion (round 6)
    picks[ff1] = 5
    picks[ff2] = 5
    picks[champion] = 6

    return {"picks": picks, "champion": champion}


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_bracket(picks: dict[str, int], truth: dict[str, int]) -> int:
    """
    Score a bracket against actual (or simulated) tournament results.

    A pick is correct if the team advances AT LEAST as far as predicted.
    Points are awarded per round per the ESPN scoring table.

    Args:
        picks: team → predicted round reached.
        truth: team → actual round reached (from simulation).

    Returns:
        Integer ESPN bracket score.
    """
    total = 0
    for team, predicted_round in picks.items():
        actual_round = truth.get(team, 0)
        for rnd in range(1, predicted_round + 1):
            if actual_round >= rnd:
                total += ESPN_ROUND_POINTS[rnd]
    return total


# ── Main simulation loop ───────────────────────────────────────────────────────

def run_simulations(
    prob_matrix: pd.DataFrame,
    n_sims: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Run N Monte Carlo bracket simulations.

    Args:
        prob_matrix: Win probability matrix from Layer 1.
        n_sims:      Number of simulations.
        seed:        Random seed for reproducibility.

    Returns:
        Dict with keys:
          - brackets: list of N simulated bracket dicts
          - scores_per_sim: (N, N) array — scores[i,j] = bracket i scored vs sim j
          - champions: list of N champion names
          - p90_bracket: bracket maximizing 90th percentile score
          - p90_scores: per-bracket p90 scores
    """
    rng = np.random.default_rng(seed)
    log.info(f"Running {n_sims:,} simulations...")

    brackets = [sim_bracket(prob_matrix, rng) for _ in range(n_sims)]
    log.info("Simulations complete. Scoring brackets...")

    picks_list = [b["picks"] for b in brackets]
    champions = [b["champion"] for b in brackets]

    # Score each bracket against a random sample of 1,000 simulated outcomes.
    # Full N×N (100M ops, ~400MB) is unnecessary — 1k sample gives a p90
    # estimate within ~2 percentile points, sufficient to rank brackets.
    scoring_sample = 1_000
    sample_idxs = rng.choice(n_sims, size=scoring_sample, replace=False)
    sample_truths = [brackets[int(j)]["picks"] for j in sample_idxs]

    log.info(f"Scoring {n_sims:,} brackets against {scoring_sample} sampled outcomes...")
    scores = np.zeros((n_sims, scoring_sample), dtype=np.int32)
    for j, truth in enumerate(sample_truths):
        for i in range(n_sims):
            scores[i, j] = score_bracket(picks_list[i], truth)

    # p90 score for each bracket = 90th percentile across the 1,000 sampled outcomes
    p90_scores = np.percentile(scores, 90, axis=1)
    best_idx = int(np.argmax(p90_scores))

    log.info(f"Best bracket index: {best_idx} (p90 score: {p90_scores[best_idx]:.0f})")

    return {
        "brackets": brackets,
        "scores": scores,
        "champions": champions,
        "p90_bracket": brackets[best_idx],
        "p90_scores": p90_scores,
        "best_idx": best_idx,
    }


# ── Output helpers ─────────────────────────────────────────────────────────────

def top_champions(champions: list[str], n: int = 5) -> pd.DataFrame:
    """
    Return top N most common champion picks with frequency.

    Args:
        champions: List of champion names from all simulations.
        n:         Number of top picks to return.

    Returns:
        DataFrame with TEAM and FREQUENCY_PCT columns.
    """
    counts = pd.Series(champions).value_counts()
    top = counts.head(n).reset_index()
    top.columns = ["TEAM", "COUNT"]
    top["FREQUENCY_PCT"] = (top["COUNT"] / len(champions) * 100).round(1)
    return top[["TEAM", "FREQUENCY_PCT"]]


def plot_score_distribution(scores: np.ndarray, best_idx: int, out_path: Path) -> None:
    """
    Plot distribution of p90 scores across all simulated brackets.

    Args:
        scores:    (N, N) score matrix.
        best_idx:  Index of the best bracket.
        out_path:  Path to save the PNG.
    """
    # Mean score per bracket across the scoring sample
    mean_scores = scores.mean(axis=1)
    best_score = mean_scores[best_idx]
    p90_scores = np.percentile(scores, 90, axis=1)
    best_p90 = p90_scores[best_idx]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: mean score distribution
    axes[0].hist(mean_scores, bins=60, color="steelblue", alpha=0.8, edgecolor="white")
    axes[0].axvline(best_score, color="crimson", lw=2, linestyle="--", label=f"Best bracket: {best_score:.0f}")
    axes[0].set_xlabel("Mean Score Across All Simulations", fontsize=11)
    axes[0].set_ylabel("Number of Brackets", fontsize=11)
    axes[0].set_title("Distribution of Mean Bracket Scores\n(10,000 simulations)", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Right: p90 score distribution
    axes[1].hist(p90_scores, bins=60, color="seagreen", alpha=0.8, edgecolor="white")
    axes[1].axvline(best_p90, color="crimson", lw=2, linestyle="--", label=f"Best bracket p90: {best_p90:.0f}")
    axes[1].set_xlabel("90th Percentile Score", fontsize=11)
    axes[1].set_ylabel("Number of Brackets", fontsize=11)
    axes[1].set_title("Distribution of P90 Bracket Scores\n(optimization target)", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    log.info(f"Score distribution plot saved to {out_path}")


def print_bracket(bracket: dict) -> None:
    """
    Pretty-print a bracket's picks grouped by the furthest round each team reached.

    picks[team] = furthest round reached (1=R64 winner ... 6=Champion).
    Each round label shows the teams that advanced TO that round (i.e. won round N-1
    and are being picked to win round N or further).
    """
    picks = bracket["picks"]
    round_names = {1: "Round of 64", 2: "Round of 32", 3: "Sweet 16",
                   4: "Elite 8", 5: "Final Four", 6: "Champion"}

    # Group teams by furthest round — each team appears exactly once
    # at their maximum predicted round
    by_round: dict[int, list[str]] = {r: [] for r in range(1, 7)}
    for team, rnd in picks.items():
        by_round[rnd].append(team)

    print("\nBest Bracket (P90-maximizing):")
    print("=" * 45)
    for rnd in range(1, 7):
        teams = sorted(by_round[rnd])
        label = round_names[rnd]
        pts = ESPN_ROUND_POINTS[rnd]
        n = len(teams)
        print(f"\n{label} — {n} team(s) ({pts} pts each):")
        for t in teams:
            print(f"  {t}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    prob_matrix = pd.read_csv(PROCESSED_DIR / "win_prob_matrix_2025.csv", index_col=0)

    results = run_simulations(prob_matrix, n_sims=10_000)

    # Champion frequency
    print("\nTop 5 Most Common Champion Picks:")
    print("=" * 35)
    print(top_champions(results["champions"]).to_string(index=False))

    # Best bracket summary
    print_bracket(results["p90_bracket"])

    # Score distribution plot
    plot_score_distribution(
        results["scores"],
        results["best_idx"],
        PROCESSED_DIR / "score_distribution.png",
    )

    # Summary stats
    mean_scores = results["scores"].mean(axis=1)
    p90_scores  = results["p90_scores"]
    print(f"\nScore Distribution Summary (across 1,000-sample scoring):")
    print(f"  Mean of mean scores:  {mean_scores.mean():.1f}")
    print(f"  Median mean score:    {np.median(mean_scores):.1f}")
    print(f"  Std dev:              {mean_scores.std():.1f}")
    print(f"  Best bracket mean:    {mean_scores[results['best_idx']]:.1f}")
    print(f"  Best bracket p90:     {p90_scores[results['best_idx']]:.1f}")
