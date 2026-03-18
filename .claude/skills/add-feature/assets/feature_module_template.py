"""
FEATURE_NAME.py — Compute FEATURE_COL feature.

Domain rationale:
  [Explain WHY this feature predicts tournament success. Reference SKILLS.md
   if relevant. E.g.: "Teams with high bench depth outlast starters in deep runs."]

Data source:
  [Where does the raw data come from? KenPom? Torvik? CBB dataset? Kaggle?]

Input:  UPSTREAM_FILE (in data/processed/)
Output: features_FEATURE_NAME.csv (adds FEATURE_COL column)

Known limitations:
  [List years where data may be missing, and what fallback is used.]
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import PROCESSED_DIR  # Import all paths/constants from config — never redefine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── Core computation ──────────────────────────────────────────────────────────

def compute_FEATURE_COL(
    col_a: pd.Series,
    col_b: pd.Series,
) -> pd.Series:
    """
    Compute FEATURE_COL from raw columns.

    [Describe the formula and what it captures.]

    Args:
        col_a: [Description of first input column.]
        col_b: [Description of second input column.]

    Returns:
        Series with FEATURE_COL values. NaN where inputs are missing.
    """
    # TODO: implement computation
    # Example: return col_a / col_b.replace(0, np.nan)
    raise NotImplementedError("Replace this with actual computation.")


# ── Feature builder ───────────────────────────────────────────────────────────

def build_FEATURE_NAME_features(
    upstream_path: Path = PROCESSED_DIR / "UPSTREAM_FILE",
) -> pd.DataFrame:
    """
    Add FEATURE_COL to the feature matrix.

    Loads UPSTREAM_FILE, computes FEATURE_COL, returns enriched DataFrame.

    Args:
        upstream_path: Path to UPSTREAM_FILE.

    Returns:
        DataFrame with all upstream columns + FEATURE_COL added.
    """
    if not upstream_path.exists():
        raise FileNotFoundError(f"Upstream features not found: {upstream_path}. "
                                f"Run the upstream feature builder first.")

    df = pd.read_csv(upstream_path)
    log.info(f"Loaded {len(df)} rows from {upstream_path.name}")

    # Compute feature
    df["FEATURE_COL"] = compute_FEATURE_COL(df["RAW_COL_A"], df["RAW_COL_B"])

    # Coverage report (only meaningful for tournament teams)
    tourney = df[df["SEED"].notna() & (df["YEAR"] != 2020)]
    null_pct = tourney["FEATURE_COL"].isna().mean() * 100
    if null_pct > 20:
        log.warning(f"FEATURE_COL: {null_pct:.1f}% null in tournament teams — check data source")
    else:
        log.info(f"FEATURE_COL: {null_pct:.1f}% null in {len(tourney)} tournament team-seasons")

    # Describe range for sanity check
    vals = tourney["FEATURE_COL"].dropna()
    if len(vals):
        log.info(f"  Range: [{vals.min():.3f}, {vals.max():.3f}], mean={vals.mean():.3f}")

    return df


# ── Persistence ───────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, filename: str = "features_FEATURE_NAME.csv") -> Path:
    """Save feature matrix to data/processed/."""
    out = PROCESSED_DIR / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info(f"Saved to {out}")
    return out


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = build_FEATURE_NAME_features()
    save(df)

    # Sanity-check summary on tournament teams
    tourney = df[df["SEED"].notna() & (df["YEAR"] != 2020)]
    print(f"\nShape (all): {df.shape}")
    print(f"Tournament team-seasons: {len(tourney)}")
    print(f"\nTop 10 by FEATURE_COL:")
    print(
        tourney.nlargest(10, "FEATURE_COL")[
            ["YEAR", "TEAM", "SEED", "FEATURE_COL"]
        ].to_string(index=False)
    )
