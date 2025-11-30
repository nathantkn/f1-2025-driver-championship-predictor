"""
Add finishing position as target variable to base.csv
"""
import pandas as pd
import numpy as np

print("Adding position target to base.csv...")

# Load data
base = pd.read_csv('data-cleaned/base.csv')
results = pd.read_csv('data-raw/results.csv')

print(f"Original base.csv: {len(base)} rows")

# Get position from results
results_position = results[['raceId', 'driverId', 'constructorId', 'grid', 'position']].copy()

# Convert position to numeric (handling 'R', 'D', 'W', etc. for DNF/disqualified)
results_position['position'] = pd.to_numeric(results_position['position'], errors='coerce')

# Merge position into base
base_with_position = base.merge(
    results_position,
    on=['raceId', 'driverId', 'constructorId', 'grid'],
    how='left',
    suffixes=('', '_new')
)

# If there are duplicates, keep the first non-null position
if 'position_new' in base_with_position.columns:
    base_with_position['target_position'] = base_with_position['position_new'].fillna(base_with_position.get('position', np.nan))
    base_with_position = base_with_position.drop(columns=['position_new'])
elif 'position' in base_with_position.columns:
    base_with_position = base_with_position.rename(columns={'position': 'target_position'})

# Keep target_bucket for reference but use target_position as primary target
if 'target_position' not in base_with_position.columns:
    print("ERROR: Could not add target_position column")
    exit(1)

# Remove rows where target_position is null (can't train without target)
print(f"Rows with null target_position: {base_with_position['target_position'].isna().sum()}")
base_with_position = base_with_position[base_with_position['target_position'].notna()].copy()

# Convert position to int
base_with_position['target_position'] = base_with_position['target_position'].astype(int)

print(f"Final base.csv: {len(base_with_position)} rows")
print(f"\nPosition distribution:")
print(base_with_position['target_position'].value_counts().sort_index().head(20))

# Save
base_with_position.to_csv('data-cleaned/base.csv', index=False)
print("\n✅ Updated base.csv with target_position column")
