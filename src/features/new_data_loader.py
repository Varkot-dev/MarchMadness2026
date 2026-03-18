"""
new_data_loader.py — Load and merge the 2026 MarchMadness pre-cleaned dataset.

Replaces the old features_coaching.csv pipeline entirely. Team names are
consistent across all source files — join directly on YEAR + TEAM, no mapping needed.

Primary source: KenPom Barttorvik.csv (2008–2026, 68 tournament teams per year)
Enriched with:
  - EvanMiya.csv   — KILLSHOTS, ROSTER RANK, INJURY RANK, RELATIVE RATING (2011–2026)
  - Resumes.csv    — NET RPI, Q1/Q2 wins, ELO, B POWER (2008–2026)

ROUND encoding in Tournament Matchups.csv:
  64 → lost R64    (0 wins, rounds_won=0)
  32 → lost R32    (1 win,  rounds_won=1)
  16 → lost S16    (2 wins, rounds_won=2)
   8 → lost E8     (3 wins, rounds_won=3)
   4 → lost F4     (4 wins, rounds_won=4)
   2 → runner-up   (5 wins, rounds_won=5)
   1 → champion    (6 wins, rounds_won=6)
   0 → not played  (2026 bracket — future games)

Output saved to data/processed/features_new.csv

Usage:
    python3 -m src.features.new_data_loader
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from config import PROCESSED_DIR

log = logging.getLogger(__name__)

DATA_DIR = Path("2026 MarchMadness")

# ROUND value → rounds_won (0–6 scale used throughout the model)
ROUND_TO_WINS = {64: 0, 32: 1, 16: 2, 8: 3, 4: 4, 2: 5, 1: 6, 0: None}

# Feature columns to keep from each source (drop rank columns to reduce noise)
KB_FEATURE_COLS = [
    "YEAR", "TEAM", "SEED", "CONF",
    # KenPom
    "KADJ EM", "KADJ O", "KADJ D", "KADJ T",
    # Barttorvik
    "BADJ EM", "BADJ O", "BADJ D", "BARTHAG",
    # Four factors
    "EFG%", "EFG%D", "FTR", "FTRD", "TOV%", "TOV%D", "OREB%", "DREB%",
    # Shooting
    "2PT%", "2PT%D", "3PT%", "3PT%D", "2PTR", "3PTR",
    # Roster
    "EXP", "TALENT", "AVG HGT", "EFF HGT",
    # Efficiency per possession
    "PPPO", "PPPD",
    # Schedule
    "WAB", "ELITE SOS",
    # Pace
    "BADJ T",
]

EM_FEATURE_COLS = [
    "YEAR", "TEAM",
    "RELATIVE RATING", "KILLSHOTS PER GAME", "KILL SHOTS CONCEDED PER GAME",
    "KILLSHOTS MARGIN", "ROSTER RANK", "INJURY RANK",
]

RESUME_FEATURE_COLS = [
    "YEAR", "TEAM",
    "NET RPI", "ELO", "B POWER", "Q1 W", "Q2 W", "Q1 PLUS Q2 W", "Q3 Q4 L",
    "RESUME", "R SCORE",
]


def load_new_features(
    years: list[int] | None = None,
    data_dir: Path = DATA_DIR,
    save: bool = True,
) -> pd.DataFrame:
    """
    Load and merge all new data sources into a single feature DataFrame.

    Joins KenPom+Barttorvik (primary) with EvanMiya and Resumes on YEAR+TEAM.
    Adds ROUNDS_WON column (0–6) derived from ROUND encoding.
    2026 teams get ROUNDS_WON=None (bracket not yet played).

    Args:
        years:    Filter to specific years. None = all available (2008–2026).
        data_dir: Path to the '2026 MarchMadness' folder.
        save:     If True, write to data/processed/features_new.csv.

    Returns:
        DataFrame with YEAR, TEAM, SEED, ROUNDS_WON, and all feature columns.
    """
    # ── Load primary source ────────────────────────────────────────────────────
    kb_path = data_dir / "KenPom Barttorvik.csv"
    if not kb_path.exists():
        raise FileNotFoundError(f"Primary data not found: {kb_path}")

    kb = pd.read_csv(kb_path, low_memory=False)
    log.info(f"Loaded KenPom+Barttorvik: {len(kb)} rows, years {kb['YEAR'].min()}–{kb['YEAR'].max()}")

    if years:
        kb = kb[kb["YEAR"].isin(years)]
        log.info(f"Filtered to years {years}: {len(kb)} rows")

    # Keep only needed columns (gracefully skip any missing)
    kb_cols = [c for c in KB_FEATURE_COLS if c in kb.columns]
    df = kb[kb_cols].copy()

    # Add ROUNDS_WON from ROUND column
    df["ROUNDS_WON"] = kb["ROUND"].map(ROUND_TO_WINS)

    # ── Join EvanMiya ──────────────────────────────────────────────────────────
    em_path = data_dir / "EvanMiya.csv"
    if em_path.exists():
        em = pd.read_csv(em_path, low_memory=False)
        em_cols = [c for c in EM_FEATURE_COLS if c in em.columns]
        em = em[em_cols].drop_duplicates(["YEAR", "TEAM"])
        df = df.merge(em, on=["YEAR", "TEAM"], how="left")
        matched = df["RELATIVE RATING"].notna().sum()
        log.info(f"Joined EvanMiya: {matched}/{len(df)} rows matched ({matched/len(df)*100:.1f}%)")
    else:
        log.warning(f"EvanMiya.csv not found at {em_path}, skipping")

    # ── Join Resumes ───────────────────────────────────────────────────────────
    res_path = data_dir / "Resumes.csv"
    if res_path.exists():
        res = pd.read_csv(res_path, low_memory=False)
        res_cols = [c for c in RESUME_FEATURE_COLS if c in res.columns]
        res = res[res_cols].drop_duplicates(["YEAR", "TEAM"])
        df = df.merge(res, on=["YEAR", "TEAM"], how="left")
        matched = df["NET RPI"].notna().sum()
        log.info(f"Joined Resumes: {matched}/{len(df)} rows matched ({matched/len(df)*100:.1f}%)")
    else:
        log.warning(f"Resumes.csv not found at {res_path}, skipping")

    # ── Compute SEED_DIVERGENCE ────────────────────────────────────────────────
    # SEED_DIVERGENCE = expected_seed_from_kadj_em - actual_seed (clipped ±8)
    # Positive = underseeded (KenPom rates them better than committee did).
    # Computed per year: rank all tournament teams by KADJ EM, map rank → expected seed.
    # Expected seed: rank 1-4 → seed 1, rank 5-8 → seed 2, ..., rank 61-68 → seed 16.
    seed_divs = []
    for year, grp in df.groupby("YEAR"):
        if grp["KADJ EM"].isna().all() or grp["SEED"].isna().all():
            grp = grp.copy()
            grp["SEED_DIVERGENCE"] = 0.0
        else:
            grp = grp.copy().sort_values("KADJ EM", ascending=False)
            n = len(grp)
            grp["_rank"] = range(1, n + 1)
            # Map rank to expected seed: ceil(rank / 4), capped at 16
            grp["_exp_seed"] = grp["_rank"].apply(lambda r: min(int(np.ceil(r / 4)), 16))
            raw_div = grp["SEED"] - grp["_exp_seed"]  # actual - implied; positive = underseeded
            grp["SEED_DIVERGENCE"] = raw_div.clip(-8, 8)
            grp = grp.drop(columns=["_rank", "_exp_seed"])
        seed_divs.append(grp)
    df = pd.concat(seed_divs).sort_values(["YEAR", "SEED"]).reset_index(drop=True)
    log.info(f"Computed SEED_DIVERGENCE for all years (clipped ±8)")

    # ── Clean up ───────────────────────────────────────────────────────────────
    df = df.sort_values(["YEAR", "SEED"]).reset_index(drop=True)

    log.info(f"Final dataset: {len(df)} rows, {df.shape[1]} columns")
    log.info(f"  Years: {sorted(df['YEAR'].unique())}")
    log.info(f"  Tournament teams per year: {df.groupby('YEAR').size().to_dict()}")
    log.info(f"  2026 teams: {len(df[df['YEAR']==2026])}")

    if save:
        out = PROCESSED_DIR / "features_new.csv"
        df.to_csv(out, index=False)
        log.info(f"Saved to {out}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = load_new_features()

    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample 2026 top teams:")
    print(df[df["YEAR"] == 2026][["TEAM", "SEED", "KADJ EM", "EXP", "TALENT", "WAB"]]
          .sort_values("KADJ EM", ascending=False).head(10).to_string(index=False))
    print(f"\nNaN counts (2013–2024 training rows):")
    train = df[df["YEAR"].between(2013, 2024)]
    nan_pct = (train.isna().sum() / len(train) * 100).round(1)
    print(nan_pct[nan_pct > 0].to_string())
