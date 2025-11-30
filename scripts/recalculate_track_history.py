#!/usr/bin/env python3
"""
Recalculate track history rolling features (drv_track_prev_starts and drv_track_prev_avg_points)
to properly account for sprint races.
"""

import pandas as pd
import numpy as np

print("="*70)
print("RECALCULATING TRACK HISTORY FEATURES")
print("="*70)

# Load data
base = pd.read_csv('data-cleaned/base.csv')

print(f"\nTotal rows: {len(base)}")

# Points mappings
STANDARD_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
SPRINT_POINTS = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}

def get_points(pos, is_sprint):
    pos = int(pos)
    return SPRINT_POINTS.get(pos, 0) if is_sprint else STANDARD_POINTS.get(pos, 0)

# Add points_scored column for calculations
base['points_scored'] = base.apply(
    lambda row: get_points(row['target_position'], row['is_sprint_weekend']), 
    axis=1
)

# Reset track history columns
base['drv_track_prev_starts'] = 0
base['drv_track_prev_avg_points'] = 0.0

print("\nRecalculating track history for each entry...")

# Sort by year, round, sprint order, driver
base = base.sort_values(['year', 'round', 'is_sprint_weekend', 'driverId'], 
                        ascending=[True, True, False, True]).reset_index(drop=True)

# Process each row
for idx in range(len(base)):
    if idx % 1000 == 0:
        print(f"  Processing row {idx}/{len(base)}...")
    
    row = base.iloc[idx]
    driverId = row['driverId']
    circuitId = row['circuitId']
    current_date = pd.to_datetime(row['date'])
    
    # Get all previous races for this driver at this circuit
    previous_at_track = base[
        (base['driverId'] == driverId) &
        (base['circuitId'] == circuitId) &
        (pd.to_datetime(base['date']) < current_date)
    ]
    
    if len(previous_at_track) > 0:
        base.at[idx, 'drv_track_prev_starts'] = len(previous_at_track)
        base.at[idx, 'drv_track_prev_avg_points'] = previous_at_track['points_scored'].mean()

# Round the average points
base['drv_track_prev_avg_points'] = base['drv_track_prev_avg_points'].round(2)

# Drop the temporary points_scored column
base = base.drop(columns=['points_scored'])

print(f"\n✅ Recalculated track history features")

# Save
base.to_csv('data-cleaned/base.csv', index=False)
print("✅ Saved updated base.csv")

# Show sample
print("\nSample track history stats (2025 drivers at Qatar - circuitId 78):")
qatar_2025 = base[(base['year'] == 2025) & (base['circuitId'] == 78)]
sample = qatar_2025.groupby('driverId').first().head(5)
print(sample[['drv_track_prev_starts', 'drv_track_prev_avg_points']].to_string())

print("\n" + "="*70)
print("COMPLETE")
print("="*70)
