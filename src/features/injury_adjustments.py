"""
injury_adjustments.py — Apply pre-tournament injury penalties to 2026 team features.

Injury data sourced from team reports as of Selection Sunday 2026.
Penalties are applied as additive deltas to WAB, TALENT, and KADJ O.

Penalty calibration rationale:
  - WAB range is roughly -13 to +14. An elite starter being out costs ~1.5-2.5 WAB points.
  - TALENT range is 0-91 (stars). A key rotation player costs ~4-12 TALENT points.
  - KADJ O range is 101-132. An offensive creator being out costs ~1-3 KADJ O points.
  - Severity tiers:
      "out"          → 100% of penalty applied
      "questionable" → 50% of penalty applied (expected value)

Usage:
    from src.features.injury_adjustments import apply_injury_adjustments
    df_2026 = apply_injury_adjustments(df_2026)
"""

import logging
from typing import NamedTuple

import pandas as pd

log = logging.getLogger(__name__)

# ── Injury definitions ────────────────────────────────────────────────────────
# Each entry: (player, team, status, wab_delta, talent_delta, kadj_o_delta)
#
# Status: "out" = full penalty, "questionable" = 50% penalty.
# Deltas are NEGATIVE (injuries reduce features).
#
# Sources: team injury reports, beat reporters, Selection Sunday updates.

class InjuryEntry(NamedTuple):
    player: str
    team: str
    status: str          # "out" or "questionable"
    wab_delta: float     # raw delta (negative)
    talent_delta: float  # raw delta (negative)
    kadj_o_delta: float  # raw delta (negative)


INJURIES_2026: list[InjuryEntry] = [
    # Texas Tech — JT Toppin (ACL, out for tournament)
    # 6'9" PF, top-10 prospect, team's primary interior scorer and rebounder.
    # Eliminates a starter: full penalty.
    InjuryEntry("JT Toppin",   "Texas Tech",  "out",          -2.0, -10.0, -2.5),

    # Duke — Caleb Foster (questionable, back injury)
    # Starting PG, primary ball-handler. Out or limited = massive playmaking loss.
    InjuryEntry("Caleb Foster", "Duke",        "questionable", -1.5,  -6.0, -2.0),

    # Duke — Patrick Ngongba (questionable, ankle)
    # Backup big, less critical. Half-weight because already partial role.
    InjuryEntry("Patrick Ngongba", "Duke",     "questionable", -0.5,  -2.0, -0.5),

    # Alabama — Collins Onyejiaka (out, knee)
    # Starter, key low-post presence.
    InjuryEntry("Collins Onyejiaka", "Alabama", "out",         -1.0,  -5.0, -1.0),

    # Alabama — Aden Holloway (suspended, indefinite)
    # Starting guard, important playmaker and scorer.
    InjuryEntry("Aden Holloway", "Alabama",    "out",          -1.2,  -5.0, -1.5),

    # Alabama — Keitenn Bristow (questionable, ankle)
    InjuryEntry("Keitenn Bristow", "Alabama",  "questionable", -0.5,  -2.0, -0.5),

    # Alabama — Davion Hannah (questionable, hamstring)
    InjuryEntry("Davion Hannah", "Alabama",    "questionable", -0.4,  -1.5, -0.4),

    # Gonzaga — Braden Huff (out first two games, lower body)
    # Key 7-footer, rim anchor. Full penalty for Round 1-2 impact (survival rounds).
    InjuryEntry("Braden Huff",  "Gonzaga",     "out",          -1.0,  -3.5, -1.0),

    # Kentucky — Jayden Quaintance (likely out, knee)
    # Top-5 recruit, starting PF. Already replaced him in projections but his
    # NBA-caliber athleticism can't be replicated by a backup.
    InjuryEntry("Jayden Quaintance", "Kentucky", "out",        -1.5, -10.0, -1.5),

    # Louisville — Mikel Brown Jr. (out first weekend)
    # Key guard, significant contributor to their offense.
    InjuryEntry("Mikel Brown Jr.", "Louisville", "out",        -1.0,  -4.0, -1.5),
]


# ── Core function ─────────────────────────────────────────────────────────────

def apply_injury_adjustments(
    df: pd.DataFrame,
    year: int = 2026,
    injuries: list[InjuryEntry] | None = None,
) -> pd.DataFrame:
    """
    Apply injury penalties to team features for a given tournament year.

    For each injury entry:
      - "out"          → applies 100% of the delta
      - "questionable" → applies 50% of the delta (expected-value approach)

    Modifies WAB, TALENT, and KADJ O in-place (on a copy).

    Args:
        df:       Features DataFrame with YEAR, TEAM, WAB, TALENT, KADJ O columns.
        year:     Tournament year to apply adjustments to.
        injuries: List of InjuryEntry objects. Defaults to INJURIES_2026.

    Returns:
        New DataFrame with adjusted features for the given year.
    """
    if injuries is None:
        injuries = INJURIES_2026

    df = df.copy()
    mask_year = df["YEAR"] == year

    adjustment_log: list[dict] = []

    for inj in injuries:
        mask_team = (df["TEAM"] == inj.team) & mask_year
        if not mask_team.any():
            log.warning(f"Injury entry: team '{inj.team}' not found in {year} data — skipping")
            continue

        factor = 1.0 if inj.status == "out" else 0.5

        wab_adj    = inj.wab_delta    * factor
        talent_adj = inj.talent_delta * factor
        ko_adj     = inj.kadj_o_delta * factor

        if "WAB" in df.columns:
            df.loc[mask_team, "WAB"]     = df.loc[mask_team, "WAB"]     + wab_adj
        if "TALENT" in df.columns:
            df.loc[mask_team, "TALENT"]  = df.loc[mask_team, "TALENT"]  + talent_adj
        if "KADJ O" in df.columns:
            df.loc[mask_team, "KADJ O"]  = df.loc[mask_team, "KADJ O"]  + ko_adj

        adjustment_log.append({
            "player":   inj.player,
            "team":     inj.team,
            "status":   inj.status,
            "factor":   factor,
            "WAB":      round(wab_adj, 2),
            "TALENT":   round(talent_adj, 2),
            "KADJ_O":   round(ko_adj, 2),
        })
        log.info(
            f"Injury [{inj.status}] {inj.player} ({inj.team}): "
            f"WAB{wab_adj:+.2f}  TALENT{talent_adj:+.2f}  KADJ_O{ko_adj:+.2f}"
        )

    # Log team-level totals
    teams_affected = {inj.team for inj in injuries}
    for team in sorted(teams_affected):
        row_before = df[df["YEAR"] == year].copy()  # already modified
        row_t = row_before[row_before["TEAM"] == team]
        if not row_t.empty:
            row = row_t.iloc[0]
            log.info(
                f"  {team} post-injury: WAB={row.get('WAB', 'N/A'):.2f}  "
                f"TALENT={row.get('TALENT', 'N/A'):.2f}  "
                f"KADJ_O={row.get('KADJ O', 'N/A'):.2f}"
            )

    log.info(f"Applied {len(adjustment_log)} injury adjustments to {len(teams_affected)} teams")
    return df
