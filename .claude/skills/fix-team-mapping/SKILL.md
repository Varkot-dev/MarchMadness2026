# Skill: fix-team-mapping

**Description:** Use when team name mismatches cause NaN features for tournament teams. Diagnoses which CBB names are unmapped to Kaggle TeamIDs and patches `CBB_TO_KAGGLE_NAMES` in `src/features/coaching.py`.

**Triggers:** "fix team mapping", "unmapped teams", "team name mismatch", "missing teams", "NaN features for tournament teams"

---

## What This Skill Does

Bridges the CBB dataset (barttorvik.com spelling) and Kaggle dataset (their own spelling) by finding missing entries in `CBB_TO_KAGGLE_NAMES` and adding the correct Kaggle team names.

---

## Steps

### 1. Diagnose which teams are unmapped

```python
python3 -c "
import pandas as pd
from src.features.coaching import CBB_TO_KAGGLE_NAMES
from src.utils.team_names import build_kaggle_to_cbb_map

feats = pd.read_csv('data/processed/features_coaching.csv')
teams = pd.read_csv('data/external/kaggle/MTeams.csv')

# Find tournament teams that have no mapping
tourney = feats[feats['SEED'].notna()]
cbb_names = tourney['TEAM'].unique()

kaggle_name_to_id = teams.set_index('TeamName')['TeamID'].to_dict()

unmapped = []
for name in cbb_names:
    kaggle_name = CBB_TO_KAGGLE_NAMES.get(name, name)
    if kaggle_name not in kaggle_name_to_id:
        unmapped.append(name)

print(f'Unmapped tournament teams: {len(unmapped)}')
for name in sorted(unmapped):
    # Fuzzy match against Kaggle names
    words = name.replace('.','').replace(\"'\", '').split()[:2]
    candidates = [t for t in kaggle_name_to_id if any(w.lower() in t.lower() for w in words)][:4]
    print(f'  {name!r:40s} -> candidates: {candidates}')
"
```

### 2. Verify the correct Kaggle name

Look up by exact name to confirm the right TeamID:

```python
python3 -c "
import pandas as pd
teams = pd.read_csv('data/external/kaggle/MTeams.csv')
# Replace 'Query' with the candidate name
print(teams[teams['TeamName'].str.contains('Query', case=False)][['TeamID','TeamName']])
"
```

### 3. Add entries to CBB_TO_KAGGLE_NAMES

Open `src/features/coaching.py` and add entries to the `CBB_TO_KAGGLE_NAMES` dict (line ~45):

```python
"CBB Name":    "Exact Kaggle TeamName",
```

Follow the existing alphabetical ordering. The CBB name is what appears in `features_coaching.csv` TEAM column. The Kaggle name must exactly match a `TeamName` in `MTeams.csv`.

### 4. Verify coverage improved

Re-run the feature that was missing and check coverage:

```python
python3 -c "
import pandas as pd
from src.features.coaching import CBB_TO_KAGGLE_NAMES
feats = pd.read_csv('data/processed/features_coaching.csv')
teams = pd.read_csv('data/external/kaggle/MTeams.csv')
kaggle_name_to_id = teams.set_index('TeamName')['TeamID'].to_dict()
tourney = feats[feats['SEED'].notna()]
total = tourney['TEAM'].nunique()
mapped = sum(1 for t in tourney['TEAM'].unique() if CBB_TO_KAGGLE_NAMES.get(t, t) in kaggle_name_to_id)
print(f'Coverage: {mapped}/{total} ({mapped/total*100:.1f}%)')
"
```

---

## Key Files

- `src/features/coaching.py` — `CBB_TO_KAGGLE_NAMES` dict is the single source of truth
- `data/external/kaggle/MTeams.csv` — authoritative list of Kaggle team names
- `data/processed/features_coaching.csv` — CBB team names (TEAM column)

## Common Mismatches Pattern

| CBB name pattern | Kaggle name pattern |
|---|---|
| `St.` (abbrev) | `St` (no period) |
| `Saint X` | `St X` |
| `Northern X` | `N X` |
| `Eastern X` | `E X` |
| `X State` | `X St` |
| `Wisconsin-X` | `WI X` |
| `UNC X` | `UNC X` (same) |
| `Louisiana Lafayette` | `Louisiana` |
| `Illinois Chicago` | `IL Chicago` |
