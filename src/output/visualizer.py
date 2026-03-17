"""
visualizer.py — NCAA bracket visualizer.

Renders a traditional bracket layout with 4 regions arranged around a center,
showing predicted picks vs actual outcomes color-coded:
  - Green:  correctly predicted (picked team actually advanced to that round)
  - Red:    incorrectly predicted (picked team was eliminated before that round)
  - Gray:   team not picked to advance this far (neutral slot)

Layout:
  - South (top-left)    East (bottom-left)
  - West  (top-right)   Midwest (bottom-right)
  - Final Four and Championship in the center

Usage:
    python -m src.output.visualizer --year 2024 --bracket-file path/to/bracket.csv
    python -m src.output.visualizer --year 2024 --simulate
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Layout constants ───────────────────────────────────────────────────────────

# Figure dimensions (width, height in inches at 100 dpi → 1600×900 px)
FIG_WIDTH = 16.0
FIG_HEIGHT = 9.0

# Each team slot box dimensions (in axes-fraction units)
SLOT_W = 0.095   # width of one team box
SLOT_H = 0.030   # height of one team box
SLOT_PAD = 0.005 # vertical padding between paired slots

# Horizontal positions for each round column (left half, mirrored for right half)
# Rounds 1–4 are regional. Round 5 = F4, Round 6 = Championship.
LEFT_ROUND_X = [0.02, 0.12, 0.22, 0.32]    # South/East: left-to-right
RIGHT_ROUND_X = [0.98, 0.88, 0.78, 0.68]   # West/Midwest: right-to-left
CENTER_F4_X_LEFT = 0.42    # South/East F4 game slot (feeds to center)
CENTER_F4_X_RIGHT = 0.58   # West/Midwest F4 game slot (feeds to center)
CHAMPIONSHIP_X = 0.50      # dead center

# Colors
COLOR_CORRECT = "#4caf50"    # green
COLOR_WRONG = "#f44336"      # red
COLOR_NEUTRAL = "#e0e0e0"    # light gray
COLOR_TEXT = "#212121"       # near-black text
COLOR_HEADER = "#1565c0"     # dark blue for round headers

# Region positions: (region_name, side)
# side = "left" uses LEFT_ROUND_X; side = "right" uses RIGHT_ROUND_X
# top/bottom determines vertical placement within the half
REGION_CONFIG = [
    {"name": "South",   "side": "left",  "half": "top"},
    {"name": "East",    "side": "left",  "half": "bottom"},
    {"name": "West",    "side": "right", "half": "top"},
    {"name": "Midwest", "side": "right", "half": "bottom"},
]

ROUND_LABELS = {1: "R64", 2: "R32", 3: "S16", 4: "E8", 5: "F4", 6: "Champ"}


# ── Core rendering helpers ─────────────────────────────────────────────────────

def _pick_color(team: str, predicted_round: int, actual: dict[str, int]) -> str:
    """
    Determine slot color based on whether the pick was correct.

    Args:
        team:            Team name.
        predicted_round: Round the model predicted this team would reach.
        actual:          Actual results dict (team → furthest round reached).

    Returns:
        Hex color string.
    """
    if not actual:
        return COLOR_NEUTRAL
    actual_round = actual.get(team, 0)
    if actual_round >= predicted_round:
        return COLOR_CORRECT
    return COLOR_WRONG


def _draw_team_slot(
    ax: plt.Axes,
    x: float,
    y: float,
    team: str,
    seed: int | None,
    color: str,
    align_right: bool = False,
) -> None:
    """
    Draw a single team slot box at the given axes-fraction coordinates.

    Args:
        ax:          Matplotlib Axes.
        x:           Left edge of slot (or right edge if align_right=True).
        y:           Bottom edge of slot.
        team:        Team name (truncated to ~15 chars if needed).
        seed:        Seed number, or None if unknown.
        color:       Background fill color.
        align_right: If True, x is the right edge and box extends left.
    """
    if align_right:
        box_x = x - SLOT_W
    else:
        box_x = x

    box = FancyBboxPatch(
        (box_x, y), SLOT_W, SLOT_H,
        boxstyle="round,pad=0.003",
        facecolor=color,
        edgecolor="#bdbdbd",
        linewidth=0.5,
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(box)

    # Truncate team name
    display_name = (team[:14] + "…") if len(team) > 15 else team
    seed_str = f"{seed} " if seed is not None else ""
    label = f"{seed_str}{display_name}"

    text_x = box_x + SLOT_W / 2
    text_y = y + SLOT_H / 2

    ax.text(
        text_x, text_y, label,
        ha="center", va="center",
        fontsize=4.5,
        color=COLOR_TEXT,
        fontweight="bold" if color == COLOR_CORRECT else "normal",
        transform=ax.transAxes,
        zorder=3,
        clip_on=True,
    )


def _draw_connector_line(
    ax: plt.Axes,
    x1: float, y1: float,
    x2: float, y2: float,
) -> None:
    """Draw a thin connector line between two slots in axes coordinates."""
    ax.plot(
        [x1, x2], [y1, y2],
        color="#9e9e9e", linewidth=0.4, zorder=1,
        transform=ax.transAxes,
    )


# ── Region layout computation ──────────────────────────────────────────────────

def _region_slot_positions(
    half: str,
    n_teams: int = 16,
) -> dict[int, list[float]]:
    """
    Compute vertical (y) center positions for each team slot in each round.

    The bracket compresses geometrically: R64 has 16 slots, R32 has 8, etc.
    Slots are evenly distributed vertically within the assigned half of the figure.

    Args:
        half:    "top" (y in [0.52, 0.97]) or "bottom" (y in [0.03, 0.48]).
        n_teams: Number of first-round teams (16 for a standard region).

    Returns:
        Dict mapping round_number → list of slot bottom-y positions.
    """
    if half == "top":
        y_min, y_max = 0.52, 0.97
    else:
        y_min, y_max = 0.03, 0.48

    available_height = y_max - y_min
    n_rounds = int(np.log2(n_teams))  # = 4 for a 16-team region

    positions: dict[int, list[float]] = {}
    for rnd in range(1, n_rounds + 1):
        n_slots = n_teams // (2 ** (rnd - 1))
        slot_step = available_height / n_slots
        centers = [
            y_min + slot_step * (i + 0.5) - SLOT_H / 2
            for i in range(n_slots)
        ]
        positions[rnd] = centers

    return positions


# ── Main bracket rendering ─────────────────────────────────────────────────────

def render_bracket(
    picks: dict[str, int],
    bracket_structure: dict,
    actual: dict[str, int] | None = None,
    year: int | None = None,
    out_path: Path | None = None,
) -> Path:
    """
    Render a full NCAA tournament bracket visualization to a PNG file.

    Each team slot is color-coded:
      - Green:  picked team actually advanced to that round.
      - Red:    picked team was eliminated before that round.
      - Gray:   no actual results provided (preview mode).

    Args:
        picks:             Dict of team_name → furthest predicted round (1–6).
                           Output format from sim_bracket()["picks"].
        bracket_structure: Output of build_bracket_from_seeds() — provides
                           regional matchup order for layout.
        actual:            Dict of team_name → actual furthest round reached.
                           If None, all slots are rendered gray (no scoring).
        year:              Tournament year for title and filename.
        out_path:          Override save path.  Defaults to
                           data/processed/bracket_visualization_{year}.png.

    Returns:
        Path where the PNG was saved.
    """
    if out_path is None:
        year_label = str(year) if year else "unknown"
        out_path = PROCESSED_DIR / f"bracket_visualization_{year_label}.png"

    actual = actual or {}

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    year_label = str(year) if year else ""
    actual_champ = _find_champion(actual) if actual else ""
    pred_champ = _find_champion(picks)

    title = f"NCAA Tournament Bracket {year_label}"
    if actual_champ:
        title += f"  |  Actual: {actual_champ}  |  Predicted: {pred_champ}"
    else:
        title += f"  |  Predicted champion: {pred_champ}"

    ax.text(
        0.5, 0.995, title,
        ha="center", va="top",
        fontsize=9, fontweight="bold",
        color=COLOR_HEADER,
        transform=ax.transAxes,
    )

    # Round column header labels (top of figure)
    round_label_y = 0.985
    for rnd_idx, rnd in enumerate(range(1, 5)):
        lbl = ROUND_LABELS[rnd]
        ax.text(
            LEFT_ROUND_X[rnd_idx] + SLOT_W / 2, round_label_y,
            lbl, ha="center", va="top",
            fontsize=5.5, color=COLOR_HEADER, fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            RIGHT_ROUND_X[rnd_idx] - SLOT_W / 2, round_label_y,
            lbl, ha="center", va="top",
            fontsize=5.5, color=COLOR_HEADER, fontweight="bold",
            transform=ax.transAxes,
        )

    for label, x in [("F4", CENTER_F4_X_LEFT), ("F4", CENTER_F4_X_RIGHT),
                      ("Champ", CHAMPIONSHIP_X)]:
        ax.text(
            x, round_label_y,
            label, ha="center", va="top",
            fontsize=5.5, color=COLOR_HEADER, fontweight="bold",
            transform=ax.transAxes,
        )

    # Region name labels
    for cfg in REGION_CONFIG:
        rname = cfg["name"]
        half = cfg["half"]
        side = cfg["side"]
        label_y = 0.975 if half == "top" else 0.495
        label_x = LEFT_ROUND_X[0] if side == "left" else RIGHT_ROUND_X[0]
        ha = "left" if side == "left" else "right"
        ax.text(
            label_x, label_y, rname,
            ha=ha, va="top",
            fontsize=6.5, fontweight="bold",
            color="#424242",
            transform=ax.transAxes,
        )

    # Build seed lookup from picks + bracket structure
    seed_lookup = _build_seed_lookup(bracket_structure)

    # Draw each region
    for cfg in REGION_CONFIG:
        rname = cfg["name"]
        half = cfg["half"]
        side = cfg["side"]
        round_xs = LEFT_ROUND_X if side == "left" else RIGHT_ROUND_X
        align_right = side == "right"

        region_matchups = bracket_structure.get("regions", {}).get(rname, [])
        if not region_matchups:
            log.warning(f"No matchup data for region {rname} — skipping")
            continue

        # Get all teams in this region (in first-round order)
        region_teams_ordered = _get_region_teams_in_order(
            region_matchups, bracket_structure.get("first_four", [])
        )

        slot_positions = _region_slot_positions(half, n_teams=len(region_teams_ordered))

        _draw_region(
            ax=ax,
            region_name=rname,
            teams_ordered=region_teams_ordered,
            picks=picks,
            actual=actual,
            seed_lookup=seed_lookup,
            slot_positions=slot_positions,
            round_xs=round_xs,
            align_right=align_right,
        )

    # Draw Final Four and Championship center section
    _draw_final_four_center(ax, picks, actual, seed_lookup)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_CORRECT, edgecolor="#bdbdbd", label="Correct pick"),
        mpatches.Patch(facecolor=COLOR_WRONG, edgecolor="#bdbdbd", label="Incorrect pick"),
        mpatches.Patch(facecolor=COLOR_NEUTRAL, edgecolor="#bdbdbd", label="No result data"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        fontsize=6,
        framealpha=0.8,
    )

    plt.tight_layout(pad=0.1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Bracket visualization saved to {out_path}")
    return out_path


def _get_region_teams_in_order(
    matchups: list[tuple],
    first_four: list[tuple],
) -> list[str]:
    """
    Flatten the 8 first-round matchups into an ordered list of 16 team names.

    First Four slot_keys are resolved to the first team in the pair (for
    layout purposes — the winner is determined during simulation).

    Args:
        matchups:    8 first-round matchup tuples for a region.
        first_four:  First Four tuple list from bracket_structure.

    Returns:
        Ordered list of 16 team names (or slot_keys resolved to first team).
    """
    ff_resolution: dict[str, str] = {}
    for team_a, team_b, slot_key in first_four:
        ff_resolution[slot_key] = team_a  # show first team as placeholder

    teams: list[str] = []
    for team_a, team_b in matchups:
        teams.append(ff_resolution.get(team_a, team_a))
        teams.append(ff_resolution.get(team_b, team_b))
    return teams


def _build_seed_lookup(bracket_structure: dict) -> dict[str, int]:
    """
    Build a team_name → seed_number dict from the bracket structure.

    Attempts to parse seed from region matchup order using SEED_PAIRINGS.
    Returns an empty dict if seed info cannot be inferred.

    Args:
        bracket_structure: Output of build_bracket_from_seeds().

    Returns:
        Dict mapping team name → seed number.
    """
    from src.models.simulator import SEED_PAIRINGS

    seed_lookup: dict[str, int] = {}
    ff_resolution: dict[str, str] = {}
    for team_a, team_b, slot_key in bracket_structure.get("first_four", []):
        ff_resolution[slot_key] = team_a

    for _region, matchups in bracket_structure.get("regions", {}).items():
        for matchup_idx, (team_a, team_b) in enumerate(matchups):
            if matchup_idx < len(SEED_PAIRINGS):
                high_seed, low_seed = SEED_PAIRINGS[matchup_idx]
                real_a = ff_resolution.get(team_a, team_a)
                real_b = ff_resolution.get(team_b, team_b)
                seed_lookup[real_a] = high_seed
                seed_lookup[real_b] = low_seed

    return seed_lookup


def _draw_region(
    ax: plt.Axes,
    region_name: str,
    teams_ordered: list[str],
    picks: dict[str, int],
    actual: dict[str, int],
    seed_lookup: dict[str, int],
    slot_positions: dict[int, list[float]],
    round_xs: list[float],
    align_right: bool,
) -> None:
    """
    Draw all rounds for one region.

    Args:
        ax:             Matplotlib Axes.
        region_name:    Region label (for logging).
        teams_ordered:  16 first-round teams in bracket order.
        picks:          Model picks (team → round).
        actual:         Actual results (team → round).
        seed_lookup:    team → seed number.
        slot_positions: Round → list of slot y-positions (bottom of box).
        round_xs:       X positions for rounds 1–4 in this region.
        align_right:    Whether boxes should be right-aligned at the given x.
    """
    n_rounds = len(round_xs)

    # Round 1: draw all 16 teams
    for slot_idx, team in enumerate(teams_ordered):
        predicted_round = picks.get(team, 0)
        if predicted_round >= 1:
            color = _pick_color(team, 1, actual)
        else:
            color = COLOR_NEUTRAL

        ys = slot_positions.get(1, [])
        if slot_idx >= len(ys):
            continue

        seed = seed_lookup.get(team)
        _draw_team_slot(
            ax, round_xs[0], ys[slot_idx], team, seed, color,
            align_right=align_right,
        )

    # Rounds 2–4: draw winners (teams with picks.round >= that round)
    for rnd in range(2, n_rounds + 1):
        ys = slot_positions.get(rnd, [])
        prev_ys = slot_positions.get(rnd - 1, [])
        n_slots = len(ys)
        x = round_xs[rnd - 1]

        for slot_idx in range(n_slots):
            # The winner of this slot came from two previous slots
            prev_pair = (slot_idx * 2, slot_idx * 2 + 1)
            # Find which picked team advances to this slot
            # A team is in this round's slot if picks[team] >= rnd and they
            # were seeded into this sub-bracket pod
            team = _find_round_winner(
                slot_idx, rnd, teams_ordered, picks
            )

            if not team:
                color = COLOR_NEUTRAL
                display = "?"
                seed = None
            else:
                predicted_round = picks.get(team, 0)
                if predicted_round >= rnd:
                    color = _pick_color(team, rnd, actual)
                else:
                    color = COLOR_NEUTRAL
                display = team
                seed = seed_lookup.get(team)

            if slot_idx >= len(ys):
                continue

            _draw_team_slot(
                ax, x, ys[slot_idx],
                display if team else "—",
                seed if team else None,
                color,
                align_right=align_right,
            )


def _find_round_winner(
    slot_idx: int,
    rnd: int,
    teams_ordered: list[str],
    picks: dict[str, int],
) -> str | None:
    """
    Find which team (from picks) occupies the given slot in the given round.

    In a standard bracket, a team's slot in round N is determined by their
    position in the initial team ordering divided by 2^(N-1).

    Args:
        slot_idx:      Index of the slot in round N (0-based).
        rnd:           Round number (2–4 for regional rounds).
        teams_ordered: The 16 initial teams in first-round order.
        picks:         Model picks (team → furthest round reached).

    Returns:
        Team name that occupies this slot, or None if not determinable.
    """
    pod_size = 2 ** (rnd - 1)    # how many original teams feed into one slot
    start = slot_idx * pod_size
    end = start + pod_size
    pod_teams = teams_ordered[start:end]

    # Among teams in this pod, find the one with the highest predicted round
    # that is >= current round (i.e., the predicted winner of this pod)
    candidates = [
        (picks.get(t, 0), t) for t in pod_teams if picks.get(t, 0) >= rnd
    ]
    if not candidates:
        return None

    # Return the team predicted to advance furthest
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _draw_final_four_center(
    ax: plt.Axes,
    picks: dict[str, int],
    actual: dict[str, int],
    seed_lookup: dict[str, int],
) -> None:
    """
    Draw the Final Four and Championship slots in the center of the bracket.

    Final Four: two game slots (one on each side of center, rounds 5).
    Championship: one slot at the very center (round 6).

    Args:
        ax:          Matplotlib Axes.
        picks:       Model picks (team → round).
        actual:      Actual results (team → round).
        seed_lookup: team → seed number.
    """
    # Teams predicted to reach round 5 (Final Four)
    f4_teams = [t for t, r in picks.items() if r >= 5]
    champion = _find_champion(picks)

    # Final Four game 1 (left side — South vs East)
    f4_left_teams = f4_teams[:2] if len(f4_teams) >= 2 else f4_teams + ["—"] * (2 - len(f4_teams))
    # Final Four game 2 (right side — West vs Midwest)
    f4_right_teams = f4_teams[2:4] if len(f4_teams) >= 4 else f4_teams[2:] + ["—"] * max(0, 4 - len(f4_teams))

    f4_y_top = 0.52
    f4_y_bottom = 0.52 - SLOT_H - SLOT_PAD

    for i, team in enumerate(f4_left_teams):
        y = f4_y_top if i == 0 else f4_y_bottom
        color = _pick_color(team, 5, actual) if team != "—" else COLOR_NEUTRAL
        seed = seed_lookup.get(team)
        _draw_team_slot(
            ax, CENTER_F4_X_LEFT - SLOT_W / 2, y, team, seed, color,
            align_right=False,
        )

    for i, team in enumerate(f4_right_teams):
        y = f4_y_top if i == 0 else f4_y_bottom
        color = _pick_color(team, 5, actual) if team != "—" else COLOR_NEUTRAL
        seed = seed_lookup.get(team)
        _draw_team_slot(
            ax, CENTER_F4_X_RIGHT - SLOT_W / 2, y, team, seed, color,
            align_right=False,
        )

    # Championship slot (center)
    champ_y = (f4_y_top + f4_y_bottom) / 2
    champ_color = _pick_color(champion, 6, actual) if champion and champion != "—" else COLOR_NEUTRAL
    seed = seed_lookup.get(champion)

    _draw_team_slot(
        ax, CHAMPIONSHIP_X - SLOT_W / 2, champ_y,
        champion or "—", seed, champ_color,
        align_right=False,
    )

    # Trophy label
    ax.text(
        CHAMPIONSHIP_X, champ_y + SLOT_H + 0.015,
        "CHAMPION",
        ha="center", va="bottom",
        fontsize=7, fontweight="bold",
        color=COLOR_HEADER,
        transform=ax.transAxes,
    )


def _find_champion(results: dict[str, int]) -> str:
    """Return the team with round 6 (Champion) from a picks/results dict."""
    for team, rnd in results.items():
        if rnd == 6:
            return team
    # Fallback: return team with highest round
    if results:
        return max(results, key=lambda t: results[t])
    return "Unknown"


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render NCAA bracket visualization."
    )
    parser.add_argument("--year", type=int, required=True, help="Tournament year.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--bracket-file",
        type=str,
        help="CSV file with team,round columns (picks).",
    )
    source.add_argument(
        "--simulate",
        action="store_true",
        help="Simulate bracket from win probability matrix.",
    )
    parser.add_argument(
        "--actual-file",
        type=str,
        default=None,
        help="Optional CSV with actual team,round results for color coding.",
    )
    args = parser.parse_args()

    year = args.year

    if args.simulate:
        # Load bracket structure and prob matrix, run quick simulation
        from config import EXTERNAL_DIR
        from src.models.simulator import build_bracket_from_seeds, run_simulations

        seeds_path = EXTERNAL_DIR / "kaggle" / "MNCAATourneySeeds.csv"
        teams_path = EXTERNAL_DIR / "kaggle" / "MTeams.csv"
        matrix_path = PROCESSED_DIR / f"win_prob_matrix_{year}.csv"

        if not seeds_path.exists() or not matrix_path.exists():
            raise FileNotFoundError(
                f"Seeds file or prob matrix not found for year {year}. "
                "Run win_probability.py first."
            )

        seeds_df = pd.read_csv(seeds_path)
        teams_df = pd.read_csv(teams_path)
        prob_matrix = pd.read_csv(matrix_path, index_col=0)

        bracket_structure = build_bracket_from_seeds(seeds_df, teams_df, year)
        results = run_simulations(prob_matrix, bracket_structure, n_sims=1_000)
        picks = results["p90_bracket"]["picks"]
    else:
        # Load picks from CSV
        bracket_df = pd.read_csv(args.bracket_file)
        if "team" not in bracket_df.columns or "round" not in bracket_df.columns:
            raise ValueError("bracket-file must have 'team' and 'round' columns")
        picks = dict(zip(bracket_df["team"], bracket_df["round"]))

        # Build bracket structure (needed for layout)
        from config import EXTERNAL_DIR
        from src.models.simulator import build_bracket_from_seeds

        seeds_path = EXTERNAL_DIR / "kaggle" / "MNCAATourneySeeds.csv"
        teams_path = EXTERNAL_DIR / "kaggle" / "MTeams.csv"

        if not seeds_path.exists():
            raise FileNotFoundError(f"Seeds file not found: {seeds_path}")

        seeds_df = pd.read_csv(seeds_path)
        teams_df = pd.read_csv(teams_path)
        bracket_structure = build_bracket_from_seeds(seeds_df, teams_df, year)

    # Load actual results if provided
    actual: dict[str, int] = {}
    if args.actual_file:
        actual_df = pd.read_csv(args.actual_file)
        if "team" not in actual_df.columns or "round" not in actual_df.columns:
            raise ValueError("actual-file must have 'team' and 'round' columns")
        actual = dict(zip(actual_df["team"], actual_df["round"]))
        log.info(f"Actual results loaded: {len(actual)} teams")

    out_path = render_bracket(
        picks=picks,
        bracket_structure=bracket_structure,
        actual=actual if actual else None,
        year=year,
    )
    print(f"Bracket visualization saved to: {out_path}")
