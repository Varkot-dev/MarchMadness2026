"""
candidates.py — Engineer all candidate features for SHAP-guided selection.

Takes features_coaching.csv (raw + existing engineered columns) and produces
features_candidates.csv with ~25 candidate features including:

  Raw (previously unused):
    BARTHAG, WAB, SOS_NETRTG, NCSOS_NETRTG, LUCK, EFG_O, EFG_D, TORD, DRB,
    FTR, FTRD, ADJ_T, 3P_O, 3P_D

  Ratio / combination features:
    EFF_RATIO        = ADJOE / ADJDE           — efficiency ratio (vs raw difference)
    SHOOT_EDGE       = EFG_O - EFG_D           — shooting advantage
    TURNOVER_EDGE    = TORD - TOR              — who forces/causes more turnovers
    REBOUND_SHARE    = ORB / (ORB + DRB)       — offensive rebound share
    OFF_VERSATILITY  = 3P_O * FTR              — can they score multiple ways?
    WIN_RATE         = W / G                   — simple win percentage

  Interaction features (encode "dangerous matchup" directly):
    SEED_X_DIVERGENCE   = log(SEED) * SEED_DIVERGENCE — underseeded + high seed = danger
    COACH_X_DIVERGENCE  = COACH_PREMIUM * SEED_DIVERGENCE — elite coach + underseeded
    TQS_X_SOS           = TRUE_QUALITY_SCORE * SOS_NETRTG — quality earned vs hard schedule

  Conference tournament momentum:
    CONF_TOURNEY_WINS   — wins in conference tournament that year (0 if data missing)
    CONF_TOURNEY_CHAMPION — 1 if won conf tourney outright, else 0

Output: data/processed/features_candidates.csv
  Same row structure as features_coaching.csv (one row per team-year).
  Tournament-only rows (SEED not null, YEAR != 2020) are the usable set.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(ROOT))

from config import PROCESSED_DIR, EXTERNAL_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

KAGGLE_DIR = EXTERNAL_DIR / "kaggle"


# ── Conference tournament feature engineering ─────────────────────────────────

def _load_conf_tourney_wins(teams_path: Path, conf_path: Path) -> pd.DataFrame:
    """
    Compute per-team conference tournament wins and champion flag per season.

    Counts wins from MConferenceTourneyGames.csv using Kaggle TeamIDs,
    then joins to team names via MTeams.csv.

    Args:
        teams_path: Path to MTeams.csv (Kaggle team ID → name).
        conf_path:  Path to MConferenceTourneyGames.csv.

    Returns:
        DataFrame with columns [TEAM_KAGGLE, YEAR, CONF_TOURNEY_WINS,
        CONF_TOURNEY_CHAMP] — one row per team-season with conference
        tourney participation. Teams with no games get 0 wins.
    """
    if not conf_path.exists():
        log.warning(f"Conference tourney file not found: {conf_path}")
        return pd.DataFrame(columns=["TeamID", "YEAR", "CONF_TOURNEY_WINS", "CONF_TOURNEY_CHAMP"])

    conf = pd.read_csv(conf_path)
    conf = conf.rename(columns={"Season": "YEAR"})

    # Identify the last day of each conference tournament (conference champion won that game)
    last_day = conf.groupby(["YEAR", "ConfAbbrev"])["DayNum"].max().reset_index()
    last_day["is_final"] = True
    conf = conf.merge(last_day, on=["YEAR", "ConfAbbrev", "DayNum"], how="left")
    conf["is_final"] = conf["is_final"].fillna(False)

    # Count wins per team per season
    wins = (
        conf.groupby(["YEAR", "WTeamID"])
        .size()
        .reset_index(name="CONF_TOURNEY_WINS")
        .rename(columns={"WTeamID": "TeamID"})
    )

    # Champion = won the final game of their conference tournament
    champs = (
        conf[conf["is_final"]][["YEAR", "WTeamID"]]
        .copy()
        .rename(columns={"WTeamID": "TeamID"})
    )
    champs["CONF_TOURNEY_CHAMP"] = 1

    result = wins.merge(champs, on=["YEAR", "TeamID"], how="left")
    result["CONF_TOURNEY_CHAMP"] = result["CONF_TOURNEY_CHAMP"].fillna(0).astype(int)

    log.info(
        f"Conference tourney: {len(result)} team-season records, "
        f"{result['CONF_TOURNEY_CHAMP'].sum()} champions"
    )
    return result


def _build_kaggle_name_map(teams_path: Path) -> dict[int, str]:
    """
    Build Kaggle TeamID → team name mapping from MTeams.csv.

    Args:
        teams_path: Path to MTeams.csv.

    Returns:
        Dict {team_id: team_name}.
    """
    if not teams_path.exists():
        log.warning(f"MTeams.csv not found: {teams_path}")
        return {}
    teams = pd.read_csv(teams_path)
    return dict(zip(teams["TeamID"], teams["TeamName"]))


# ── Core feature engineering ──────────────────────────────────────────────────

def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ratio and combination features that capture relationships between
    existing columns more precisely than raw differences.

    Args:
        df: DataFrame with raw CBB + KenPom columns.

    Returns:
        df with new columns added in-place.
    """
    # Efficiency ratio: ADJOE/ADJDE — captures the multiplicative interaction.
    # A team with ADJOE=120, ADJDE=90 has ratio 1.33; one with 110/100 has 1.10.
    # The difference (30 vs 10) and ratio (1.33 vs 1.10) convey different things.
    df["EFF_RATIO"] = df["ADJOE"] / df["ADJDE"].replace(0, np.nan)

    # Shooting edge: EFG_O - EFG_D
    # Positive = team shoots better than they allow — core winning formula
    df["SHOOT_EDGE"] = df["EFG_O"] - df["EFG_D"]

    # Turnover edge: TORD (opponent TOR forced) - TOR (own turnover rate)
    # Positive = force more turnovers than you give up
    df["TURNOVER_EDGE"] = df["TORD"] - df["TOR"]

    # Offensive rebound share: ORB / (ORB + DRB)
    # Controls for teams that dominate both glass ends vs just one
    denom = (df["ORB"] + df["DRB"]).replace(0, np.nan)
    df["REBOUND_SHARE"] = df["ORB"] / denom

    # Offensive versatility: 3P_O * FTR
    # Teams that shoot well from 3 AND get to the line are hardest to defend
    df["OFF_VERSATILITY"] = df["3P_O"] * df["FTR"]

    # Win rate: W/G — simpler than WAB but captures raw dominance
    df["WIN_RATE"] = df["W"] / df["G"].replace(0, np.nan)

    log.info("Added ratio features: EFF_RATIO, SHOOT_EDGE, TURNOVER_EDGE, "
             "REBOUND_SHARE, OFF_VERSATILITY, WIN_RATE")
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction terms that encode tournament-specific signal.

    These capture the "dangerous matchup" concept that domain experts use:
    an underseeded team with an elite coach on a hard schedule is a Cinderella
    candidate. The interaction is more informative than the components alone.

    Args:
        df: DataFrame with SEED, SEED_DIVERGENCE, COACH_PREMIUM,
            TRUE_QUALITY_SCORE, SOS_NETRTG columns.

    Returns:
        df with new interaction columns added in-place.
    """
    # log(SEED) * SEED_DIVERGENCE
    # Why log(SEED)? The upset value of an 11-seed beating a 6-seed is
    # much higher (in points) than a 12 beating a 5. log compresses the
    # upper end so the interaction is informative across the seed range.
    df["SEED_X_DIVERGENCE"] = np.log1p(df["SEED"].clip(1, 16)) * df["SEED_DIVERGENCE"]

    # COACH_PREMIUM * SEED_DIVERGENCE
    # An elite coach + underseeded = the classic upset formula
    df["COACH_X_DIVERGENCE"] = df["COACH_PREMIUM"] * df["SEED_DIVERGENCE"]

    # TRUE_QUALITY_SCORE * SOS_NETRTG
    # Quality earned against hard schedules is more predictive than
    # quality earned against pushovers
    df["TQS_X_SOS"] = df["TRUE_QUALITY_SCORE"] * df["SOS_NETRTG"]

    log.info("Added interaction features: SEED_X_DIVERGENCE, COACH_X_DIVERGENCE, TQS_X_SOS")
    return df


def add_conf_tourney_features(
    df: pd.DataFrame,
    conf_path: Path = KAGGLE_DIR / "MConferenceTourneyGames.csv",
    teams_path: Path = KAGGLE_DIR / "MTeams.csv",
) -> pd.DataFrame:
    """
    Join conference tournament wins and champion flag onto the feature matrix.

    Matches on team name using Kaggle's MTeams.csv name → CBB name bridge.
    Teams with no conference tournament data get 0 wins and 0 champion flag.

    Args:
        df:         Feature matrix with TEAM and YEAR columns.
        conf_path:  Path to MConferenceTourneyGames.csv.
        teams_path: Path to MTeams.csv.

    Returns:
        df with CONF_TOURNEY_WINS and CONF_TOURNEY_CHAMP added.
    """
    conf_wins = _load_conf_tourney_wins(teams_path, conf_path)
    if conf_wins.empty:
        df["CONF_TOURNEY_WINS"] = 0
        df["CONF_TOURNEY_CHAMP"] = 0
        return df

    # Build TeamID → name map for joining
    id_to_name = _build_kaggle_name_map(teams_path)
    conf_wins["TEAM_KAGGLE"] = conf_wins["TeamID"].map(id_to_name)

    # Join on Kaggle name — many CBB names match Kaggle names directly.
    # Mismatches fall back to 0 via fillna after the merge.
    merged = df.merge(
        conf_wins[["YEAR", "TEAM_KAGGLE", "CONF_TOURNEY_WINS", "CONF_TOURNEY_CHAMP"]],
        left_on=["YEAR", "TEAM"],
        right_on=["YEAR", "TEAM_KAGGLE"],
        how="left",
    )

    matched = merged["CONF_TOURNEY_WINS"].notna().sum()
    log.info(
        f"Conference tourney join: {matched}/{len(merged)} teams matched by name. "
        f"Unmatched teams get 0 wins."
    )

    merged["CONF_TOURNEY_WINS"] = merged["CONF_TOURNEY_WINS"].fillna(0)
    merged["CONF_TOURNEY_CHAMP"] = merged["CONF_TOURNEY_CHAMP"].fillna(0).astype(int)
    merged = merged.drop(columns=["TEAM_KAGGLE"], errors="ignore")

    return merged


# ── Full candidate feature build ──────────────────────────────────────────────

# All candidate columns that will be evaluated by SHAP selector.
# Includes raw-but-unused columns from features_coaching.csv,
# plus all engineered features added by this module.
CANDIDATE_FEATURES = [
    # Previously used
    "TRUE_QUALITY_SCORE",
    "SEED_DIVERGENCE",
    # Raw but previously unused
    "BARTHAG",
    "WAB",
    "SOS_NETRTG",
    "NCSOS_NETRTG",
    "LUCK",
    "EFG_O",
    "EFG_D",
    "TORD",
    "DRB",
    "FTR",
    "FTRD",
    "ADJ_T",
    "3P_O",
    "3P_D",
    "ADJOE",
    "ADJDE",
    "TOR",
    "ORB",
    # Engineered ratio / combination
    "EFF_RATIO",
    "SHOOT_EDGE",
    "TURNOVER_EDGE",
    "REBOUND_SHARE",
    "OFF_VERSATILITY",
    "WIN_RATE",
    # Interaction
    "SEED_X_DIVERGENCE",
    "COACH_X_DIVERGENCE",
    "TQS_X_SOS",
    # Conference tournament
    "CONF_TOURNEY_WINS",
    "CONF_TOURNEY_CHAMP",
]


def build_candidate_features(
    features_path: Path = PROCESSED_DIR / "features_coaching.csv",
) -> pd.DataFrame:
    """
    Full pipeline: load features_coaching.csv, add all engineered candidate
    features, return the enriched DataFrame.

    Args:
        features_path: Path to features_coaching.csv.

    Returns:
        DataFrame with all original columns + all CANDIDATE_FEATURES columns.
    """
    log.info(f"Loading base features from {features_path}")
    df = pd.read_csv(features_path)
    log.info(f"  {len(df)} rows, {df.shape[1]} columns")

    df = add_ratio_features(df)
    df = add_interaction_features(df)
    df = add_conf_tourney_features(df)

    # Coverage report for tournament teams (the only rows the model uses)
    tourney = df[df["SEED"].notna() & (df["YEAR"] != 2020)]
    log.info(f"\nCandidate feature coverage ({len(tourney)} tournament team-seasons):")
    for col in CANDIDATE_FEATURES:
        if col not in tourney.columns:
            log.warning(f"  MISSING: {col}")
            continue
        null_pct = tourney[col].isna().mean() * 100
        flag = " ⚠" if null_pct > 20 else ""
        log.info(f"  {col:<25} null={null_pct:5.1f}%{flag}")

    return df


def save(df: pd.DataFrame, path: Path = PROCESSED_DIR / "features_candidates.csv") -> Path:
    """Save candidate feature matrix to data/processed/."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"Saved candidate features to {path}")
    return path


if __name__ == "__main__":
    df = build_candidate_features()
    save(df)

    tourney = df[df["SEED"].notna() & (df["YEAR"] != 2020)]
    print(f"\nCandidate feature matrix: {tourney.shape[0]} tournament team-seasons")
    print(f"Total candidate features: {len(CANDIDATE_FEATURES)}")
    print(f"\nNew columns added:")
    base_cols = set(pd.read_csv(PROCESSED_DIR / "features_coaching.csv").columns)
    new_cols = [c for c in df.columns if c not in base_cols]
    for c in new_cols:
        vals = tourney[c].dropna()
        print(f"  {c:<25} mean={vals.mean():.3f}  std={vals.std():.3f}  null={tourney[c].isna().mean()*100:.1f}%")
