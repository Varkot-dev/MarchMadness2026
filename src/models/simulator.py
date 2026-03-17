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
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import EXTERNAL_DIR, PROCESSED_DIR, ESPN_ROUND_POINTS, SEED_PAIRINGS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Kaggle region-letter → standard region name
KAGGLE_REGION_MAP = {
    "W": "South",
    "X": "East",
    "Y": "West",
    "Z": "Midwest",
}


# ── Bracket construction ───────────────────────────────────────────────────────

def build_bracket_from_seeds(
    seeds_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    year: int,
) -> dict:
    """
    Build a bracket structure dict from Kaggle seed and teams data.

    Parses the Seed column (e.g. "W01", "X11a", "Z16b") to extract region,
    seed number, and First Four play-in flag.  Teams with the same region
    + seed number but different 'a'/'b' suffix play each other in the
    First Four.  The 16 remaining (or play-in winner) slots are then
    arranged into the 8 standard first-round matchups per region.

    Args:
        seeds_df:  DataFrame from MNCAATourneySeeds.csv.
                   Required columns: Season, Seed, TeamID.
        teams_df:  DataFrame from MTeams.csv.
                   Required columns: TeamID, TeamName.
        year:      Tournament season year to build for.

    Returns:
        Dict with keys:
          "regions"    : {region_name: [(team_a, team_b), ...]} — 8 matchups
                         per region in seed-pairing order.
          "first_four" : [(team_a, team_b, slot_key), ...] — play-in games.
                         slot_key matches the placeholder used in the regions
                         dict when one of the two teams is a First Four team.
    """
    year_seeds = seeds_df[seeds_df["Season"] == year].copy()
    if year_seeds.empty:
        raise ValueError(f"No seed data found for year {year}")

    # Map TeamID → TeamName
    id_to_name: dict[int, str] = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))

    # Parse each seed entry
    parsed: list[dict] = []
    for _, row in year_seeds.iterrows():
        seed_str = str(row["Seed"])          # e.g. "W01", "X11a", "Z16b"
        region_letter = seed_str[0]
        suffix = ""
        if seed_str[-1] in ("a", "b"):
            suffix = seed_str[-1]
            seed_num = int(seed_str[1:3])
        else:
            seed_num = int(seed_str[1:3])

        region_name = KAGGLE_REGION_MAP.get(region_letter, region_letter)
        team_id = int(row["TeamID"])
        team_name = id_to_name.get(team_id, f"Team_{team_id}")

        parsed.append({
            "region": region_name,
            "seed_num": seed_num,
            "suffix": suffix,
            "team": team_name,
            "team_id": team_id,
        })

    df = pd.DataFrame(parsed)

    # ── Identify First Four pairs ──────────────────────────────────────────────
    # A First Four pair: two entries with the same (region, seed_num) and
    # non-empty suffix ('a' and 'b').
    ff_pairs: list[tuple] = []
    first_four_teams: set[str] = set()

    grouped = df[df["suffix"] != ""].groupby(["region", "seed_num"])
    for (region, seed_num), group in grouped:
        if len(group) == 2:
            teams_in_pair = group["team"].tolist()
            # slot_key: a stable string for this play-in slot used as placeholder
            slot_key = f"FF_{region}_{seed_num}"
            ff_pairs.append((teams_in_pair[0], teams_in_pair[1], slot_key))
            first_four_teams.update(teams_in_pair)
            log.debug(
                f"First Four: {teams_in_pair[0]} vs {teams_in_pair[1]} "
                f"({region} seed {seed_num})"
            )

    # ── Build per-region matchups ──────────────────────────────────────────────
    # Only keep the 16 main-bracket entries per region.
    # For First Four seeds, use the slot_key as the team placeholder.
    # Build a seed_num → team_or_placeholder mapping per region.
    regions: dict[str, list[tuple]] = {}

    for region_name in df["region"].unique():
        region_df = df[df["region"] == region_name]

        seed_map: dict[int, str] = {}
        for _, row in region_df.iterrows():
            s = row["seed_num"]
            team = row["team"]
            suffix = row["suffix"]

            if team in first_four_teams:
                # Both 'a' and 'b' entries for this seed → use slot_key
                slot_key = f"FF_{region_name}_{s}"
                seed_map[s] = slot_key
            else:
                # Normal team (no suffix, or only one entry for this seed)
                seed_map[s] = team

        # Build the 8 standard matchups using SEED_PAIRINGS order
        matchups: list[tuple] = []
        for high_seed, low_seed in SEED_PAIRINGS:
            if high_seed not in seed_map or low_seed not in seed_map:
                log.warning(
                    f"Missing seed {high_seed} or {low_seed} in {region_name} "
                    f"for year {year} — skipping matchup"
                )
                continue
            matchups.append((seed_map[high_seed], seed_map[low_seed]))

        regions[region_name] = matchups

    return {
        "regions": regions,
        "first_four": ff_pairs,
    }


# ── Core simulation ────────────────────────────────────────────────────────────

def sim_game(
    team_a: str,
    team_b: str,
    prob_matrix: pd.DataFrame,
    rng: np.random.Generator,
) -> str:
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
    if team_a not in prob_matrix.index or team_b not in prob_matrix.index:
        # Graceful fallback: flip a fair coin when a team is missing from matrix
        log.warning(
            f"Team not in prob_matrix: '{team_a}' vs '{team_b}' — using 50/50"
        )
        return team_a if rng.random() < 0.5 else team_b
    p = prob_matrix.loc[team_a, team_b]
    return team_a if rng.random() < p else team_b


def resolve_first_four(
    bracket_structure: dict,
    prob_matrix: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, str]:
    """
    Simulate the First Four games and return slot_key → winner mapping.

    Args:
        bracket_structure: Output of build_bracket_from_seeds().
        prob_matrix:       Win probability matrix.
        rng:               NumPy random generator.

    Returns:
        Dict mapping slot_key → winning team name.
    """
    ff_slots: dict[str, str] = {}
    for team_a, team_b, slot_key in bracket_structure.get("first_four", []):
        winner = sim_game(team_a, team_b, prob_matrix, rng)
        ff_slots[slot_key] = winner
    return ff_slots


def sim_region(
    matchups: list[tuple],
    prob_matrix: pd.DataFrame,
    rng: np.random.Generator,
    ff_winners: dict[str, str],
) -> tuple[list[list[str]], str]:
    """
    Simulate one region (4 rounds: R64 → R32 → S16 → E8).

    Args:
        matchups:    8 first-round matchups for this region.
                     Entries may be slot_key strings if they are First Four slots.
        prob_matrix: Win probability matrix.
        rng:         NumPy random generator.
        ff_winners:  slot_key → winner mapping from resolve_first_four().

    Returns:
        Tuple of (round_winners_per_round, regional_champion).
        round_winners_per_round[0] = 8 R64 winners, [1] = 4 R32 winners, etc.
        The final element is a single-team list containing the E8 champion.
    """
    # Resolve any First Four placeholders
    current: list[tuple[str, str]] = []
    for team_a, team_b in matchups:
        team_a = ff_winners.get(team_a, team_a)
        team_b = ff_winners.get(team_b, team_b)
        current.append((team_a, team_b))

    all_rounds: list[list[str]] = []
    teams = current
    while len(teams) > 1:
        winners = [sim_game(a, b, prob_matrix, rng) for a, b in teams]
        all_rounds.append(winners)
        teams = list(zip(winners[::2], winners[1::2]))

    # E8 champion is the last single remaining team; record explicitly
    champion = all_rounds[-1][0]
    all_rounds.append([champion])
    return all_rounds, champion


def sim_final_four(
    region_champs: list[str],
    region_names: list[str],
    prob_matrix: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[str, str, str]:
    """
    Simulate Final Four and Championship (rounds 5 and 6).

    Traditional bracket pairing: first region vs second region, third vs fourth.
    Standard Kaggle ordering (South, East, West, Midwest) pairs South vs East
    and West vs Midwest.

    Args:
        region_champs: Champions in the same order as region_names.
        region_names:  Ordered list of region names used in simulation.
        prob_matrix:   Win probability matrix.
        rng:           NumPy random generator.

    Returns:
        Tuple of (ff_winner_1, ff_winner_2, champion).
    """
    if len(region_champs) != 4:
        raise ValueError(
            f"Expected 4 region champions, got {len(region_champs)}: {region_champs}"
        )
    r1, r2, r3, r4 = region_champs
    ff1 = sim_game(r1, r2, prob_matrix, rng)
    ff2 = sim_game(r3, r4, prob_matrix, rng)
    champion = sim_game(ff1, ff2, prob_matrix, rng)
    return ff1, ff2, champion


def sim_bracket(
    prob_matrix: pd.DataFrame,
    rng: np.random.Generator,
    bracket_structure: dict,
) -> dict:
    """
    Simulate one full 68-team tournament bracket.

    Args:
        prob_matrix:       Win probability matrix (teams as index/columns).
        rng:               NumPy random generator.
        bracket_structure: Output of build_bracket_from_seeds() — contains
                           "regions" and "first_four" keys.

    Returns:
        Dict with keys:
          - picks:    team_name → furthest round reached (1–6).
                      Round 1 = R64 winner; 6 = Champion.
          - champion: tournament champion team name.
    """
    ff_winners = resolve_first_four(bracket_structure, prob_matrix, rng)

    region_names = list(bracket_structure["regions"].keys())
    region_results: dict[str, list[list[str]]] = {}
    region_champs: list[str] = []

    for region in region_names:
        matchups = bracket_structure["regions"][region]
        rounds, champ = sim_region(matchups, prob_matrix, rng, ff_winners)
        region_results[region] = rounds
        region_champs.append(champ)

    ff1, ff2, champion = sim_final_four(
        region_champs, region_names, prob_matrix, rng
    )

    # Build picks: team → furthest round reached (1=R64 winner … 6=champion)
    picks: dict[str, int] = {}
    for region in region_names:
        rounds = region_results[region]
        for round_idx, winners in enumerate(rounds):
            round_num = round_idx + 1
            for team in winners:
                picks[team] = round_num

    # Final Four (round 5) and Champion (round 6) override any earlier entry
    picks[ff1] = 5
    picks[ff2] = 5
    picks[champion] = 6

    return {"picks": picks, "champion": champion}


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_bracket(picks: dict[str, int], truth: dict[str, int]) -> int:
    """
    Score a bracket against actual (or simulated) tournament results.

    A pick earns points for each round in which the team actually advanced at
    least as far as predicted.  Points are per the ESPN scoring table.

    Args:
        picks: team → predicted furthest round reached.
        truth: team → actual furthest round reached.

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
    bracket_structure: dict,
    n_sims: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Run N Monte Carlo bracket simulations.

    Args:
        prob_matrix:       Win probability matrix from Layer 1.
        bracket_structure: Output of build_bracket_from_seeds().
        n_sims:            Number of simulations (default 10,000).
        seed:              Random seed for reproducibility.

    Returns:
        Dict with keys:
          - brackets:     list of N simulated bracket dicts.
          - scores:       (N, scoring_sample) array — scores[i,j] = bracket i
                          scored against sampled outcome j.
          - champions:    list of N champion names.
          - p90_bracket:  bracket maximizing 90th-percentile score.
          - p90_scores:   per-bracket p90 scores array.
          - best_idx:     index of the p90-maximizing bracket.
    """
    rng = np.random.default_rng(seed)
    log.info(f"Running {n_sims:,} simulations...")

    brackets = [
        sim_bracket(prob_matrix, rng, bracket_structure) for _ in range(n_sims)
    ]
    log.info("Simulations complete. Scoring brackets...")

    picks_list = [b["picks"] for b in brackets]
    champions = [b["champion"] for b in brackets]

    # Score each bracket against a random sample of 1,000 simulated outcomes.
    # Full N×N (100M ops) is unnecessary — 1k sample gives a p90 estimate
    # within ~2 percentile points, sufficient to rank brackets.
    scoring_sample = 1_000
    sample_idxs = rng.choice(n_sims, size=scoring_sample, replace=False)
    sample_truths = [brackets[int(j)]["picks"] for j in sample_idxs]

    log.info(
        f"Scoring {n_sims:,} brackets against {scoring_sample} sampled outcomes..."
    )
    scores = np.zeros((n_sims, scoring_sample), dtype=np.int32)
    for j, truth in enumerate(sample_truths):
        for i in range(n_sims):
            scores[i, j] = score_bracket(picks_list[i], truth)

    # p90 score for each bracket = 90th percentile across the 1,000 sampled outcomes
    p90_scores = np.percentile(scores, 90, axis=1)
    best_idx = int(np.argmax(p90_scores))

    log.info(
        f"Best bracket index: {best_idx} (p90 score: {p90_scores[best_idx]:.0f})"
    )

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


def plot_score_distribution(
    scores: np.ndarray,
    best_idx: int,
    out_path: Path,
) -> None:
    """
    Plot distribution of mean and p90 scores across all simulated brackets.

    Args:
        scores:    (N, scoring_sample) score matrix.
        best_idx:  Index of the best bracket.
        out_path:  Path to save the PNG.
    """
    mean_scores = scores.mean(axis=1)
    best_score = mean_scores[best_idx]
    p90_scores = np.percentile(scores, 90, axis=1)
    best_p90 = p90_scores[best_idx]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: mean score distribution
    axes[0].hist(mean_scores, bins=60, color="steelblue", alpha=0.8, edgecolor="white")
    axes[0].axvline(
        best_score, color="crimson", lw=2, linestyle="--",
        label=f"Best bracket: {best_score:.0f}",
    )
    axes[0].set_xlabel("Mean Score Across All Simulations", fontsize=11)
    axes[0].set_ylabel("Number of Brackets", fontsize=11)
    axes[0].set_title(
        "Distribution of Mean Bracket Scores\n(10,000 simulations)", fontsize=12
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Right: p90 score distribution
    axes[1].hist(p90_scores, bins=60, color="seagreen", alpha=0.8, edgecolor="white")
    axes[1].axvline(
        best_p90, color="crimson", lw=2, linestyle="--",
        label=f"Best bracket p90: {best_p90:.0f}",
    )
    axes[1].set_xlabel("90th Percentile Score", fontsize=11)
    axes[1].set_ylabel("Number of Brackets", fontsize=11)
    axes[1].set_title(
        "Distribution of P90 Bracket Scores\n(optimization target)", fontsize=12
    )
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    log.info(f"Score distribution plot saved to {out_path}")


def print_bracket(bracket: dict) -> None:
    """
    Pretty-print a bracket's picks grouped by furthest round reached.

    picks[team] = furthest round reached (1=R64 winner ... 6=Champion).
    Each round label shows the teams that advanced TO that round.

    Args:
        bracket: Dict with "picks" key mapping team_name → round_number.
    """
    picks = bracket["picks"]
    round_names = {
        1: "Round of 64",
        2: "Round of 32",
        3: "Sweet 16",
        4: "Elite 8",
        5: "Final Four",
        6: "Champion",
    }

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
    if len(sys.argv) < 2:
        print("Usage: python simulator.py <year>")
        print("Example: python simulator.py 2024")
        sys.exit(1)

    year = int(sys.argv[1])
    kaggle_dir = EXTERNAL_DIR / "kaggle"
    seeds_path = kaggle_dir / "MNCAATourneySeeds.csv"
    teams_path = kaggle_dir / "MTeams.csv"
    matrix_path = PROCESSED_DIR / f"win_prob_matrix_{year}.csv"

    if not seeds_path.exists():
        raise FileNotFoundError(
            f"Seeds file not found: {seeds_path}\n"
            "Download from: https://www.kaggle.com/competitions/march-machine-learning-mania-2025"
        )
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Win probability matrix not found: {matrix_path}\n"
            "Run win_probability.py first to generate it."
        )

    seeds_df = pd.read_csv(seeds_path)
    teams_df = pd.read_csv(teams_path)
    prob_matrix = pd.read_csv(matrix_path, index_col=0)

    log.info(f"Building bracket structure for {year}...")
    bracket_structure = build_bracket_from_seeds(seeds_df, teams_df, year)
    log.info(
        f"Bracket built: {len(bracket_structure['regions'])} regions, "
        f"{len(bracket_structure['first_four'])} First Four games"
    )

    results = run_simulations(prob_matrix, bracket_structure, n_sims=10_000)

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
        PROCESSED_DIR / f"score_distribution_{year}.png",
    )

    # Summary stats
    mean_scores = results["scores"].mean(axis=1)
    p90_scores = results["p90_scores"]
    print(f"\nScore Distribution Summary (across 1,000-sample scoring):")
    print(f"  Mean of mean scores:  {mean_scores.mean():.1f}")
    print(f"  Median mean score:    {np.median(mean_scores):.1f}")
    print(f"  Std dev:              {mean_scores.std():.1f}")
    print(f"  Best bracket mean:    {mean_scores[results['best_idx']]:.1f}")
    print(f"  Best bracket p90:     {p90_scores[results['best_idx']]:.1f}")
