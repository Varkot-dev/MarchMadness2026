"""
config.py — All project constants, paths, and hyperparameters.
Never hardcode paths elsewhere — import from here.
"""

from pathlib import Path

# ── Root ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

# ── Data directories ──────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

# ── Raw data files ────────────────────────────────────────────────────────────
CBB_MAIN = RAW_DIR / "cbb.csv"        # 2013-2024 (excl. 2020)
CBB_2020 = RAW_DIR / "cbb20.csv"     # 2020 season (no tournament, COVID)
CBB_2025 = RAW_DIR / "cbb25.csv"     # 2025 season (current year)

# ── Train / holdout split ─────────────────────────────────────────────────────
# Per HANDOFF.md: k-fold CV on 2013-2015, holdout is 2016-2025
# Touch the holdout ONCE at the very end.
TRAIN_YEARS = list(range(2013, 2016))       # k-fold feature selection / tuning
HOLDOUT_YEARS = list(range(2016, 2026))     # final evaluation only

# ── Scoring (ESPN bracket) ────────────────────────────────────────────────────
ROUND_POINTS = {
    1: 1,   # Round of 64
    2: 2,   # Round of 32
    3: 4,   # Sweet 16
    4: 8,   # Elite 8
    5: 16,  # Final Four
    6: 32,  # Champion
}

# ── Feature engineering ───────────────────────────────────────────────────────
LUCK_COEFFICIENT = 0.4          # True Quality Score: AdjEM - (Luck * 0.4)
QMS_WEIGHTS = {                 # Quality Momentum Score opponent weights
    "top_25": 10,
    "top_50": 7,
    "top_100": 4,
    "below_100": 1,
}
PROSPECT_WEIGHTS = {            # NBA Prospect Depth Score
    "top_15": 10,
    "top_30": 7,
    "top_50": 4,
}

# ── Model ─────────────────────────────────────────────────────────────────────
MONTE_CARLO_SIMS = 10_000
RANDOM_SEED = 42
