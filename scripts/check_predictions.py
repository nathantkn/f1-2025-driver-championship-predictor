#!/usr/bin/env python3
"""
Check per-race predictions for top drivers
"""

import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('trained_knn_pipeline.joblib')

# Load remaining races
remaining = pd.read_csv('data-cleaned/remaining_races.csv')

# Load drivers for names
drivers = pd.read_csv('data-raw/drivers.csv')
driver_names = drivers[['driverId', 'forename', 'surname']].copy()
driver_names['driver_name'] = driver_names['forename'] + ' ' + driver_names['surname']

# Features used by the model
categorical_features = ['circuitId']
passthrough_features = ['is_sprint_weekend']
numeric_features = [
    'grid',
    'cum_points_before', 'position_before', 'wins_before',
    'drv_last3_avg_points', 'drv_last3_top10_rate', 'drv_last3_dnf_rate', 'drv_last3_avg_grid',
    'tm_last3_avg_points', 'tm_last3_top10_rate',
    'drv_track_prev_starts', 'drv_track_prev_avg_points'
]

# Fill missing values
for col in numeric_features:
    if col in remaining.columns:
        remaining[col] = remaining[col].fillna(0)

# Prepare features
X = remaining[categorical_features + passthrough_features + numeric_features]

# Get predictions
y_proba = model.predict_proba(X)

# Convert to expected points
target_classes = ['0', '1–3', '4–6', '7–10', '12–15', '18–26']
midpoints = {'0': 0, '1–3': 2, '4–6': 5, '7–10': 8.5, '12–15': 13.5, '18–26': 22}

expected_points = np.zeros(len(y_proba))
for i, class_name in enumerate(target_classes):
    expected_points += y_proba[:, i] * midpoints[class_name]

remaining['expected_points'] = expected_points

# Merge with driver names
remaining = remaining.merge(driver_names[['driverId', 'driver_name']], on='driverId', how='left')

# Show predictions for top drivers
top_drivers = [846, 857, 839]  # Lando, Oscar, Max
driver_map = {846: 'Lando Norris', 857: 'Oscar Piastri', 839: 'Max Verstappen'}

print("=" * 80)
print("PER-RACE PREDICTIONS FOR TOP 3 DRIVERS")
print("=" * 80)

for driver_id in top_drivers:
    driver_data = remaining[remaining['driverId'] == driver_id].copy()
    driver_data = driver_data.sort_values('round')
    
    print(f"\n{driver_map[driver_id]} (driverId={driver_id}):")
    print(f"Current points: {driver_data['cum_points_before'].iloc[0]:.0f}")
    
    for _, row in driver_data.iterrows():
        race_name = f"Round {row['round']}"
        if row['round'] == 23:
            race_name += " (Qatar Sprint)"
        elif row['round'] == 24:
            race_name += " (Qatar GP)"
        elif row['round'] == 25:
            race_name += " (Abu Dhabi GP)"
        
        print(f"  {race_name}: {row['expected_points']:.2f} points")
    
    total = driver_data['expected_points'].sum()
    print(f"  Total projected from remaining races: {total:.2f} points")
    print(f"  Final projected total: {driver_data['cum_points_before'].iloc[0] + total:.2f} points")

print("\n" + "=" * 80)
print("SUMMARY: Total entries and predictions")
print("=" * 80)
print(f"Total entries in remaining_races.csv: {len(remaining)}")
print(f"Unique drivers: {remaining['driverId'].nunique()}")
print(f"Races per driver: {len(remaining) / remaining['driverId'].nunique():.1f}")
print("\nProbability distributions for Lando Norris (first race):")
lando_first = remaining[(remaining['driverId'] == 846) & (remaining['round'] == 23)].iloc[0]
print(f"Expected points: {lando_first['expected_points']:.2f}")
