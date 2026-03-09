"""
fetch_coaching.py — Parse Bart Torvik coaching tournament data.

Input:  data/external/coaching/coaches_2013_2025.txt
        (one file, all years, tab-separated with year headers)

Output: data/processed/coaching_raw.csv
        Columns: YEAR, COACH, CONF, G, W, L
"""

import logging
import re
from pathlib import Path

import pandas as pd

from config import EXTERNAL_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

COACHING_FILE = EXTERNAL_DIR / "coaching" / "coaches_2013_2025.txt"

# Matches lines like: "1\tTom Izzo\tB10\t5\t4–1"
ROW_PATTERN = re.compile(
    r"^\d+\t(?P<coach>.+?)\t(?P<conf>\S+)\t(?P<g>\d+)\t(?P<w>\d+)[–-](?P<l>\d+)$"
)


def _parse_section(year: int, lines: list[str]) -> list[dict]:
    """
    Parse one year's worth of coaching rows.

    Args:
        year:  Season year.
        lines: Lines from that year's section.

    Returns:
        List of dicts with YEAR, COACH, CONF, G, W, L.
    """
    records = []
    for line in lines:
        line = line.strip()
        m = ROW_PATTERN.match(line)
        if not m:
            continue
        records.append({
            "YEAR": year,
            "COACH": m.group("coach").strip(),
            "CONF": m.group("conf").strip(),
            "G": int(m.group("g")),
            "W": int(m.group("w")),
            "L": int(m.group("l")),
        })
    return records


def parse_coaching_file(path: Path = COACHING_FILE) -> pd.DataFrame:
    """
    Parse the full multi-year coaching file into a clean DataFrame.

    Args:
        path: Path to the coaching text file.

    Returns:
        DataFrame with one row per coach per tournament year.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Split into per-year sections
    sections = re.split(r"(\d{4}) T-Rank and Tempo-Free Stats", content)
    # sections: ['preamble', '2013', data, '2014', data, ...]

    all_records = []
    for i in range(1, len(sections) - 1, 2):
        year = int(sections[i])
        data = sections[i + 1]
        lines = data.strip().split("\n")
        records = _parse_section(year, lines)
        all_records.extend(records)
        log.info(f"{year}: {len(records)} coaches parsed")

    df = pd.DataFrame(all_records)
    df = df.sort_values(["YEAR", "COACH"]).reset_index(drop=True)
    log.info(f"Total: {len(df)} coach-year records across years {sorted(df['YEAR'].unique())}")
    return df


def save(df: pd.DataFrame, filename: str = "coaching_raw.csv") -> Path:
    """Save parsed coaching data to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / filename
    df.to_csv(out, index=False)
    log.info(f"Saved to {out}")
    return out


if __name__ == "__main__":
    df = parse_coaching_file()
    save(df)
    print(f"\nShape: {df.shape}")
    print(f"\nSample:")
    print(df[df["COACH"] == "Tom Izzo"].to_string(index=False))
