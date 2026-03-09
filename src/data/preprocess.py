"""
preprocess.py — Load, clean, and merge all raw CBB datasets into one DataFrame.

Handles schema differences across cbb.csv (2013-2024), cbb20.csv (2020),
and cbb25.csv (2025), then writes the merged result to data/processed/.
"""

import logging
from pathlib import Path

import pandas as pd

from config import CBB_MAIN, CBB_2020, CBB_2025, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Columns to drop — not team performance related
DROP_COLS = {"RK", "3PR", "3PRD"}

# Canonical column name map (handles casing inconsistencies)
RENAME_MAP = {"Team": "TEAM"}


def load_main() -> pd.DataFrame:
    """Load cbb.csv (2013-2024, excludes 2020)."""
    df = pd.read_csv(CBB_MAIN)
    log.info(f"Loaded cbb.csv: {len(df)} rows, years {sorted(df['YEAR'].unique())}")
    return df


def load_2020() -> pd.DataFrame:
    """Load cbb20.csv and add YEAR=2020. No POSTSEASON/SEED (tournament cancelled)."""
    df = pd.read_csv(CBB_2020)
    df["YEAR"] = 2020
    df["POSTSEASON"] = None
    df["SEED"] = None
    log.info(f"Loaded cbb20.csv: {len(df)} rows, added YEAR=2020")
    return df


def load_2025() -> pd.DataFrame:
    """Load cbb25.csv and add YEAR=2025. No POSTSEASON yet (tournament upcoming)."""
    df = pd.read_csv(CBB_2025)
    df["YEAR"] = 2025
    df["POSTSEASON"] = None
    log.info(f"Loaded cbb25.csv: {len(df)} rows, added YEAR=2025")
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a DataFrame to the canonical schema:
    - Rename inconsistent column names
    - Drop non-performance columns
    - Standardize column name casing

    Args:
        df: Raw DataFrame from any of the three source files.

    Returns:
        Cleaned DataFrame with consistent columns.
    """
    df = df.rename(columns=RENAME_MAP)
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        log.info(f"Dropped columns: {cols_to_drop}")
    return df


def merge_all() -> pd.DataFrame:
    """
    Load, normalize, and merge all three source files into one DataFrame.

    Returns:
        Merged DataFrame sorted by YEAR then TEAM, with consistent schema.
    """
    frames = [normalize(load_main()), normalize(load_2020()), normalize(load_2025())]
    merged = pd.concat(frames, ignore_index=True)

    # Sort for readability
    merged = merged.sort_values(["YEAR", "TEAM"]).reset_index(drop=True)

    log.info(
        f"Merged dataset: {len(merged)} rows, "
        f"years {sorted(merged['YEAR'].unique())}, "
        f"columns: {list(merged.columns)}"
    )
    return merged


def save(df: pd.DataFrame, filename: str = "cbb_merged.csv") -> Path:
    """
    Save the merged DataFrame to data/processed/.

    Args:
        df: Merged DataFrame to save.
        filename: Output filename.

    Returns:
        Path to the saved file.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / filename
    df.to_csv(out_path, index=False)
    log.info(f"Saved merged dataset to {out_path}")
    return out_path


if __name__ == "__main__":
    df = merge_all()
    save(df)
    print(f"\nShape: {df.shape}")
    print(f"Years: {sorted(df['YEAR'].unique())}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample (2020, no tournament):")
    print(df[df["YEAR"] == 2020][["TEAM", "YEAR", "SEED", "POSTSEASON", "ADJOE", "ADJDE"]].head(3))
    print(f"\nSample (2025):")
    print(df[df["YEAR"] == 2025][["TEAM", "YEAR", "SEED", "ADJOE", "ADJDE"]].head(3))
