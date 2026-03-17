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

# ESPN bracket challenge scoring (actual point values used in competition)
ESPN_ROUND_POINTS = {
    1: 10,   # Round of 64
    2: 20,   # Round of 32
    3: 40,   # Sweet 16
    4: 80,   # Elite 8
    5: 160,  # Final Four
    6: 320,  # Championship
}

# Standard NCAA bracket first-round seed pairings (higher seed vs lower seed)
SEED_PAIRINGS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

# Region order determines Final Four pairings: South vs East, West vs Midwest
REGIONS = ["South", "East", "West", "Midwest"]

# First Four play-in games for 2025 (team_a, team_b, region, seed)
FIRST_FOUR_2025 = [
    ("VCU",            "San Diego St.",    "South",   11),
    ("Saint Francis",  "Alabama St.",      "South",   16),
    ("North Carolina", "Drake",            "East",    11),
    ("American",       "Mount St. Mary's", "East",    16),
    ("Xavier",         "Texas",            "West",    11),
    ("Norfolk St.",    "SIU Edwardsville", "West",    16),
]

# Seed Divergence clipping bounds — prevents extreme KenPom rank mismatches
# from destabilizing LR coefficients (e.g. rank 200 team seeded 1 = -49 raw)
SEED_DIVERGENCE_CLIP = (-8, 8)

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
