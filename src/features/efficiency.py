"""
efficiency.py — Compute efficiency-based features.

Features built here:
  - True Quality Score: AdjEM - (Luck * 0.4)
    Strips luck from efficiency to reveal true team strength.
  - Seed Divergence Score: KenPom implied seed - actual seed
    Positive = underseeded (sleeper), Negative = overseeded (avoid).

Produces data/processed/features_efficiency.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, LUCK_COEFFICIENT

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# KenPom rank → implied seed bin (ranks 1-4 = seed 1, 5-8 = seed 2, etc.)
# 68 tournament teams → 4 per seed line × 16 seeds + 4 play-in = ~68
def kenpom_rank_to_implied_seed(rank: float) -> float:
    """
    Convert KenPom rank to an implied tournament seed (1–16).

    Uses the standard binning: every 4 ranks = 1 seed line.
    Rank 1-4 → seed 1, rank 5-8 → seed 2, ..., rank 61-68 → seed 16.

    Args:
        rank: KenPom national rank (1 = best).

    Returns:
        Implied seed as float (1.0–16.0+), or NaN if rank is NaN.
    """
    if pd.isna(rank):
        return np.nan
    return min(16.0, np.ceil(rank / 4.0))


def compute_true_quality_score(
    adj_em: pd.Series,
    luck: pd.Series,
    coeff: float = LUCK_COEFFICIENT,
) -> pd.Series:
    """
    Compute True Quality Score = AdjEM - (Luck * coeff).

    Luck correction: teams with positive luck won more games than their
    efficiency deserved. In a single-elimination tournament, luck regresses
    to zero. Stripping it out reveals actual team strength.

    Args:
        adj_em: Adjusted efficiency margin (KenPom NetRtg).
        luck:   KenPom Luck rating.
        coeff:  Luck penalty coefficient (default 0.4, calibrate later).

    Returns:
        True Quality Score series.
    """
    return adj_em - (luck * coeff)


def compute_seed_divergence(
    kenpom_rank: pd.Series,
    actual_seed: pd.Series,
) -> pd.Series:
    """
    Compute Seed Divergence = KenPom implied seed - actual seed.

    Positive value → team is underseeded (stronger than seed suggests).
    Negative value → team is overseeded (weaker than seed suggests).

    Args:
        kenpom_rank: KenPom national rank for each team.
        actual_seed: Actual NCAA tournament seed (1–16).

    Returns:
        Seed Divergence series (NaN for non-tournament teams).
    """
    implied = kenpom_rank.apply(kenpom_rank_to_implied_seed)
    return implied - actual_seed


def _join_kenpom(cbb: pd.DataFrame, kp: pd.DataFrame) -> pd.DataFrame:
    """Left-join KenPom columns onto CBB data on YEAR + TEAM."""
    kp_cols = [
        "YEAR", "TEAM", "KENPOM_NETRTG", "KENPOM_ORTG", "KENPOM_DRTG",
        "KENPOM_ADJT", "LUCK", "SOS_NETRTG", "NCSOS_NETRTG",
    ]
    df = cbb.merge(kp[kp_cols], on=["YEAR", "TEAM"], how="left")
    matched = df["LUCK"].notna().sum()
    log.info(f"KenPom join: {matched}/{len(df)} rows matched ({len(df)-matched} missing)")
    return df


def _add_true_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add TRUE_QUALITY_SCORE; falls back to ADJOE-ADJDE for teams missing KenPom."""
    fallback_count = df["KENPOM_NETRTG"].isna().sum()
    if fallback_count:
        log.warning(f"TQS fallback (no KenPom): {fallback_count} teams using ADJOE-ADJDE")
    luck_zero_count = df["LUCK"].isna().sum()
    if luck_zero_count:
        log.warning(f"TQS fallback (no Luck): {luck_zero_count} teams using Luck=0")
    df["TRUE_QUALITY_SCORE"] = compute_true_quality_score(
        adj_em=df["KENPOM_NETRTG"].fillna(df["ADJOE"] - df["ADJDE"]),
        luck=df["LUCK"].fillna(0),
        coeff=LUCK_COEFFICIENT,
    )
    return df


def _add_seed_divergence(df: pd.DataFrame) -> pd.DataFrame:
    """Add KENPOM_RANK and SEED_DIVERGENCE columns."""
    df["KENPOM_RANK"] = df.groupby("YEAR")["KENPOM_NETRTG"].rank(
        ascending=False, method="min", na_option="bottom"
    )
    df["SEED_DIVERGENCE"] = compute_seed_divergence(
        kenpom_rank=df["KENPOM_RANK"],
        actual_seed=df["SEED"],
    )
    return df


def build_efficiency_features(
    cbb_path: Path = PROCESSED_DIR / "cbb_merged.csv",
    kenpom_path: Path = PROCESSED_DIR / "kenpom_merged.csv",
) -> pd.DataFrame:
    """
    Join CBB and KenPom data, then compute efficiency features.

    Args:
        cbb_path:    Path to merged CBB dataset.
        kenpom_path: Path to merged KenPom dataset.

    Returns:
        DataFrame with all original columns plus KenPom fields,
        TRUE_QUALITY_SCORE, and SEED_DIVERGENCE.
    """
    cbb = pd.read_csv(cbb_path)
    kp = pd.read_csv(kenpom_path)
    log.info(f"CBB: {len(cbb)} rows | KenPom: {len(kp)} rows")

    df = _join_kenpom(cbb, kp)
    df = _add_true_quality_score(df)
    df = _add_seed_divergence(df)

    log.info(f"TQS range: {df['TRUE_QUALITY_SCORE'].min():.2f} to {df['TRUE_QUALITY_SCORE'].max():.2f}")
    return df


def save(df: pd.DataFrame, filename: str = "features_efficiency.csv") -> Path:
    """Save efficiency features to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / filename
    df.to_csv(out, index=False)
    log.info(f"Saved to {out}")
    return out


if __name__ == "__main__":
    df = build_efficiency_features()
    save(df)

    tourney = df[df["SEED"].notna() & (df["YEAR"] != 2020) & (df["YEAR"] != 2025)]

    print(f"\nShape: {df.shape}")
    print(f"\nTop 10 True Quality Score (all years, tournament teams):")
    print(
        tourney.nlargest(10, "TRUE_QUALITY_SCORE")[
            ["YEAR", "TEAM", "SEED", "KENPOM_NETRTG", "LUCK", "TRUE_QUALITY_SCORE"]
        ].to_string(index=False)
    )

    print(f"\nMost underseeded teams (highest Seed Divergence):")
    print(
        tourney.nlargest(10, "SEED_DIVERGENCE")[
            ["YEAR", "TEAM", "SEED", "KENPOM_RANK", "SEED_DIVERGENCE"]
        ].to_string(index=False)
    )

    print(f"\nMost overseeded teams (lowest Seed Divergence):")
    print(
        tourney.nsmallest(10, "SEED_DIVERGENCE")[
            ["YEAR", "TEAM", "SEED", "KENPOM_RANK", "SEED_DIVERGENCE"]
        ].to_string(index=False)
    )
