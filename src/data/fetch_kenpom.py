"""
fetch_kenpom.py — Parse KenPom exported text files into a clean DataFrame.

Handles two export formats found in the raw files:
  - Tab-separated: 2013-2019, 2021 (one team per line)
  - Column-major:  2020, 2022-2025 (one value per line, repeated header blocks)

Output columns (what we need):
  YEAR, TEAM, CONF, W, L, KENPOM_NETRTG, KENPOM_ORTG, KENPOM_DRTG,
  KENPOM_ADJT, LUCK, SOS_NETRTG, NCSOS_NETRTG
"""

import logging
import re
from pathlib import Path

import pandas as pd

from config import EXTERNAL_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

KENPOM_DIR = EXTERNAL_DIR / "kenpom"

# Columns in the tab-separated header row (matches the two-row header structure)
# "Strength of Schedule  NCSOS"
# "Rk Team Conf W-L NetRtg ORtg DRtg AdjT Luck | NetRtg ORtg DRtg | NetRtg"
# We assign positional names since ranks interleave with values
TAB_COLS = [
    "RK", "TEAM", "CONF", "WL", "KENPOM_NETRTG", "KENPOM_ORTG_VAL", "KENPOM_ORTG_RK",
    "KENPOM_DRTG_VAL", "KENPOM_DRTG_RK", "KENPOM_ADJT_VAL", "KENPOM_ADJT_RK",
    "LUCK", "LUCK_RK",
    "SOS_NETRTG", "SOS_NETRTG_RK", "SOS_ORTG", "SOS_ORTG_RK", "SOS_DRTG", "SOS_DRTG_RK",
    "NCSOS_NETRTG", "NCSOS_NETRTG_RK",
]

# Column-major format repeats this sequence for each team
COL_MAJOR_FIELDS = [
    "RK", "TEAM", "CONF", "WL", "KENPOM_NETRTG", "KENPOM_ORTG_VAL", "KENPOM_ORTG_RK",
    "KENPOM_DRTG_VAL", "KENPOM_DRTG_RK", "KENPOM_ADJT_VAL", "KENPOM_ADJT_RK",
    "LUCK", "LUCK_RK",
    "SOS_NETRTG", "SOS_NETRTG_RK", "SOS_ORTG", "SOS_ORTG_RK", "SOS_DRTG", "SOS_DRTG_RK",
    "NCSOS_NETRTG", "NCSOS_NETRTG_RK",
]
N_FIELDS = len(COL_MAJOR_FIELDS)

# Header lines to skip in column-major format
# Map CBB dataset names → KenPom names (applied after parsing KenPom)
# Inverse: map KenPom names → canonical CBB names
KENPOM_TO_CBB = {
    "LIU": "LIU Brooklyn",
    "N.C. State": "North Carolina St.",
}

SKIP_PATTERNS = {
    "Strength of Schedule", "NCSOS", "Rk", "Team", "Conf", "W-L",
    "NetRtg", "ORtg", "DRtg", "AdjT", "Luck",
    "Remember me", "Forgot password", "ADVANCED ANALYSIS",
    "Data through", "© KenPom", "kenpom.com",
}


def clean_team_name(name: str) -> str:
    """Remove championship/seed markers like 'Duke 1' → 'Duke', then apply CBB name mapping."""
    cleaned = re.sub(r"\s+\d+[\*]*$", "", name.strip())
    return KENPOM_TO_CBB.get(cleaned, cleaned)


def parse_wl(wl: str) -> tuple[int, int]:
    """Parse 'W-L' string into (wins, losses). Returns (0, 0) on failure."""
    try:
        w, l = wl.strip().split("-")
        return int(w), int(l)
    except Exception:
        return 0, 0


def is_header_line(line: str) -> bool:
    """Return True if line is a column-major header/noise line to skip."""
    stripped = line.strip()
    if not stripped:
        return True
    for pat in SKIP_PATTERNS:
        if stripped.startswith(pat):
            return True
    return False


def parse_tab_separated(path: Path, year: int) -> pd.DataFrame:
    """
    Parse a tab-separated KenPom file (2013-2019, 2021 format).

    Args:
        path: Path to the .txt file.
        year: Season year to tag rows with.

    Returns:
        DataFrame with cleaned KenPom columns.
    """
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            # Skip header rows (first col is non-numeric)
            if not parts[0].strip().lstrip("+").lstrip("-").replace(".", "").isdigit():
                continue
            if len(parts) < N_FIELDS:
                continue
            rows.append(parts[:N_FIELDS])

    df = pd.DataFrame(rows, columns=COL_MAJOR_FIELDS)
    return _finalize(df, year)


def parse_column_major(path: Path, year: int) -> pd.DataFrame:
    """
    Parse a column-major KenPom file (2020, 2022-2025 format).
    One field value per line, repeating N_FIELDS per team.

    Args:
        path: Path to the .txt file.
        year: Season year to tag rows with.

    Returns:
        DataFrame with cleaned KenPom columns.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        raw_lines = [l.rstrip("\n") for l in f]

    # Filter out header/noise lines
    values = [l.strip() for l in raw_lines if not is_header_line(l)]

    # Group into chunks of N_FIELDS
    records = []
    i = 0
    while i + N_FIELDS <= len(values):
        chunk = values[i : i + N_FIELDS]
        # Validate: first field should be a rank integer
        if chunk[0].isdigit():
            records.append(chunk)
            i += N_FIELDS
        else:
            i += 1  # re-sync if misaligned

    df = pd.DataFrame(records, columns=COL_MAJOR_FIELDS)
    return _finalize(df, year)


def _finalize(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Shared cleanup: parse W-L, clean team names, select/rename final columns.

    Args:
        df: Raw parsed DataFrame.
        year: Season year.

    Returns:
        Cleaned DataFrame ready for merging.
    """
    df["TEAM"] = df["TEAM"].apply(clean_team_name)
    df[["W", "L"]] = df["WL"].apply(lambda x: pd.Series(parse_wl(x)))
    df["YEAR"] = year

    # Cast numeric columns
    for col in ["KENPOM_NETRTG", "KENPOM_ORTG_VAL", "KENPOM_DRTG_VAL",
                "KENPOM_ADJT_VAL", "LUCK", "SOS_NETRTG", "NCSOS_NETRTG"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[[
        "YEAR", "TEAM", "CONF", "W", "L",
        "KENPOM_NETRTG", "KENPOM_ORTG_VAL", "KENPOM_DRTG_VAL",
        "KENPOM_ADJT_VAL", "LUCK", "SOS_NETRTG", "NCSOS_NETRTG",
    ]].rename(columns={
        "KENPOM_ORTG_VAL": "KENPOM_ORTG",
        "KENPOM_DRTG_VAL": "KENPOM_DRTG",
        "KENPOM_ADJT_VAL": "KENPOM_ADJT",
    })


def detect_format(path: Path) -> str:
    """
    Detect whether file is tab-separated or column-major.

    Args:
        path: Path to file.

    Returns:
        'tab' or 'column_major'
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "\t" in line and len(line.split("\t")) >= 5:
                return "tab"
    return "column_major"


def load_all_kenpom() -> pd.DataFrame:
    """
    Load and merge all KenPom text files from data/external/kenpom/.

    Returns:
        Combined DataFrame for all available years, sorted by YEAR then TEAM.
    """
    files = sorted(KENPOM_DIR.glob("*.txt"), key=lambda p: p.name.lower())
    if not files:
        raise FileNotFoundError(f"No .txt files found in {KENPOM_DIR}")

    frames = []
    for path in files:
        # Extract year from filename (e.g. kempom.2013.txt, Kempom.2025.txt)
        match = re.search(r"(\d{4})", path.name)
        if not match:
            log.warning(f"Could not extract year from {path.name}, skipping")
            continue
        year = int(match.group(1))

        fmt = detect_format(path)
        log.info(f"Parsing {path.name} (year={year}, format={fmt})")

        try:
            if fmt == "tab":
                df = parse_tab_separated(path, year)
            else:
                df = parse_column_major(path, year)

            log.info(f"  → {len(df)} teams parsed")
            frames.append(df)
        except Exception as e:
            log.error(f"  Failed to parse {path.name}: {e}")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["YEAR", "TEAM"]).reset_index(drop=True)
    log.info(f"Total KenPom records: {len(merged)}, years: {sorted(merged['YEAR'].unique())}")
    return merged


def save(df: pd.DataFrame, filename: str = "kenpom_merged.csv") -> Path:
    """Save merged KenPom DataFrame to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / filename
    df.to_csv(out, index=False)
    log.info(f"Saved to {out}")
    return out


if __name__ == "__main__":
    df = load_all_kenpom()
    save(df)

    print(f"\nShape: {df.shape}")
    print(f"Years: {sorted(df['YEAR'].unique())}")
    print(f"\nSample 2025:")
    print(df[df["YEAR"] == 2025].head(5).to_string(index=False))
    print(f"\nSample 2013:")
    print(df[df["YEAR"] == 2013].head(5).to_string(index=False))
    print(f"\nLuck range: {df['LUCK'].min():.3f} to {df['LUCK'].max():.3f}")
    print(f"Null counts:\n{df.isnull().sum()}")
