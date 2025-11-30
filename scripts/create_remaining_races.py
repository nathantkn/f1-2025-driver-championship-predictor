"""
Create remaining races CSV with proper data structure
This script extracts the last known state from base.csv and creates
properly structured entries for the 3 remaining 2025 races.
"""

import pandas as pd
import numpy as np

print("Creating remaining_races.csv from base.csv data...")

# Load existing data
base = pd.read_csv('data-cleaned/base.csv')
races = pd.read_csv('data-raw/races.csv')
results = pd.read_csv('data-raw/results.csv')
constructors = pd.read_csv('data-raw/constructors.csv')

# Get ALL drivers who competed in 2025 season (use latest appearance for each driver)
drivers_2025 = base[base['year'] == 2025].copy()
# Get the most recent entry for EACH DRIVER (to get their current team)
last_round_data = drivers_2025.sort_values('round').groupby('driverId').tail(1).copy()
print(f"Found {len(last_round_data)} drivers in 2025 season")

# Get results from round 22 to update standings
results_r22 = results[results['raceId'] == 1166]
points_r22 = results_r22.groupby('driverId')['points'].sum().reset_index()
points_r22.columns = ['driverId', 'points_r22']

# Prepare driver data for remaining races
driver_data = []

for idx, row in last_round_data.iterrows():
    driverId = row['driverId']
    constructorId = row['constructorId']
    
    # Get points scored in round 22
    points_scored = points_r22[points_r22['driverId'] == driverId]['points_r22'].values
    points_scored = points_scored[0] if len(points_scored) > 0 else 0
    
    # Update cumulative points for round 23
    cum_points_r23 = row['cum_points_before'] + points_scored
    
    # Use drv_last3_avg_grid as the grid position estimate (already calculated in rolling features)
    estimated_grid = row['drv_last3_avg_grid']
    
    driver_data.append({
        'driverId': driverId,
        'constructorId': constructorId,
        'cum_points_after_r22': cum_points_r23,
        'estimated_grid': estimated_grid,
        'drv_last3_avg_points': row['drv_last3_avg_points'],
        'drv_last3_top10_rate': row['drv_last3_top10_rate'],
        'drv_last3_dnf_rate': row['drv_last3_dnf_rate'],
        'drv_last3_avg_grid': row['drv_last3_avg_grid'],
        'tm_last3_avg_points': row['tm_last3_avg_points'],
        'tm_last3_top10_rate': row['tm_last3_top10_rate'],
        'drv_track_prev_starts_qatar': 0,  # Will update below
        'drv_track_prev_avg_points_qatar': 0.0,
        'drv_track_prev_starts_abudhabi': 0,
        'drv_track_prev_avg_points_abudhabi': 0.0
    })

driver_data_df = pd.DataFrame(driver_data)

# Get track-specific data (circuitId 78=Qatar, 24=Abu Dhabi)
results_with_circuit = results.merge(races[['raceId', 'circuitId']], on='raceId')

for circuit_id, circuit_name in [(78, 'qatar'), (24, 'abudhabi')]:
    circuit_results = results_with_circuit[results_with_circuit['circuitId'] == circuit_id]
    track_stats = circuit_results.groupby('driverId').agg({
        'resultId': 'count',
        'points': 'mean'
    }).reset_index()
    track_stats.columns = ['driverId', f'drv_track_prev_starts_{circuit_name}', f'drv_track_prev_avg_points_{circuit_name}']
    
    driver_data_df = driver_data_df.merge(track_stats, on='driverId', how='left', suffixes=('', '_new'))
    driver_data_df[f'drv_track_prev_starts_{circuit_name}'] = driver_data_df[f'drv_track_prev_starts_{circuit_name}_new'].fillna(driver_data_df[f'drv_track_prev_starts_{circuit_name}']).fillna(0)
    driver_data_df[f'drv_track_prev_avg_points_{circuit_name}'] = driver_data_df[f'drv_track_prev_avg_points_{circuit_name}_new'].fillna(driver_data_df[f'drv_track_prev_avg_points_{circuit_name}']).fillna(0.0)
    driver_data_df = driver_data_df.drop(columns=[f'drv_track_prev_starts_{circuit_name}_new', f'drv_track_prev_avg_points_{circuit_name}_new'])

# Calculate championship positions after round 22
driver_data_df = driver_data_df.sort_values('cum_points_after_r22', ascending=False).reset_index(drop=True)
driver_data_df['position_after_r22'] = range(1, len(driver_data_df) + 1)

# Count wins through round 22
# Use wins_before from the latest round 22 entry in 2025
round_22_2025 = base[(base['year'] == 2025) & (base['round'] == 22)]
round_22_latest = round_22_2025.sort_values('raceId').groupby('driverId').tail(1)[['driverId', 'wins_before']]
driver_data_df = driver_data_df.merge(round_22_latest, on='driverId', how='left')
driver_data_df['wins_through_r22'] = driver_data_df['wins_before'].fillna(0)
driver_data_df = driver_data_df.drop(columns=['wins_before'])

print(f"Prepared data for {len(driver_data_df)} drivers")

# Define the 2 remaining races (removed Qatar Sprint)
remaining_races = [
    {
        'raceId': 1168,
        'round': 24,
        'circuitId': 78,
        'circuit_name': 'qatar',
        'name': 'Qatar Grand Prix',
        'date': '2025-11-30',
        'is_sprint_weekend': 0,
        'prev_raceId': 1166
    },
    {
        'raceId': 1169,
        'round': 25,
        'circuitId': 24,
        'circuit_name': 'abudhabi',
        'name': 'Abu Dhabi Grand Prix',
        'date': '2025-12-08',
        'is_sprint_weekend': 0,
        'prev_raceId': 1168
    }
]

# Create remaining races entries
remaining_entries = []

for race in remaining_races:
    for idx, driver_row in driver_data_df.iterrows():
        remaining_entries.append({
            'raceId': race['raceId'],
            'driverId': driver_row['driverId'],
            'constructorId': driver_row['constructorId'],
            'grid': driver_row['estimated_grid'],
            'year': 2025,
            'round': race['round'],
            'circuitId': race['circuitId'],
            'is_sprint_weekend': race['is_sprint_weekend'],
            'date': race['date'],
            'prev_raceId': race['prev_raceId'],
            'cum_points_before': driver_row['cum_points_after_r22'],
            'position_before': driver_row['position_after_r22'],
            'wins_before': driver_row['wins_through_r22'],
            'drv_last3_avg_points': driver_row['drv_last3_avg_points'],
            'drv_last3_top10_rate': driver_row['drv_last3_top10_rate'],
            'drv_last3_dnf_rate': driver_row['drv_last3_dnf_rate'],
            'drv_last3_avg_grid': driver_row['drv_last3_avg_grid'],
            'tm_last3_avg_points': driver_row['tm_last3_avg_points'],
            'tm_last3_top10_rate': driver_row['tm_last3_top10_rate'],
            'drv_track_prev_starts': driver_row[f'drv_track_prev_starts_{race["circuit_name"]}'],
            'drv_track_prev_avg_points': driver_row[f'drv_track_prev_avg_points_{race["circuit_name"]}']
        })

remaining_df = pd.DataFrame(remaining_entries)

# Ensure proper column order matching base.csv (minus target_bucket)
columns_order = [
    'raceId', 'driverId', 'constructorId', 'grid', 'year', 'round', 'circuitId',
    'is_sprint_weekend', 'date', 'prev_raceId', 'cum_points_before', 'position_before',
    'wins_before', 'drv_last3_avg_points', 'drv_last3_top10_rate', 'drv_last3_dnf_rate',
    'drv_last3_avg_grid', 'tm_last3_avg_points', 'tm_last3_top10_rate',
    'drv_track_prev_starts', 'drv_track_prev_avg_points'
]

remaining_df = remaining_df[columns_order]

# Save to CSV
remaining_df.to_csv('data-cleaned/remaining_races.csv', index=False)
print(f"\n✅ Created remaining_races.csv with {len(remaining_df)} entries ({len(remaining_df)//2} drivers × 2 races)")
print(f"   Columns: {len(remaining_df.columns)}")
print(f"   Races: Round 24 (Qatar GP), Round 25 (Abu Dhabi GP)")

# Also update races.csv if needed
existing_race_ids = set(races['raceId'].values)
new_races_to_add = []

for race in remaining_races:
    if race['raceId'] not in existing_race_ids:
        new_races_to_add.append({
            'raceId': race['raceId'],
            'prev_raceId': race['prev_raceId'],
            'is_sprint_weekend': race['is_sprint_weekend'],
            'year': 2025,
            'round': race['round'],
            'circuitId': race['circuitId'],
            'name': race['name'],
            'date': race['date'],
            'time': '17:00:00',
            'url': '',
            'fp1_date': '',
            'fp1_time': '',
            'fp2_date': '',
            'fp2_time': '',
            'fp3_date': '',
            'fp3_time': '',
            'quali_date': '',
            'quali_time': '',
            'sprint_date': '',
            'sprint_time': ''
        })

if new_races_to_add:
    new_races_df = pd.DataFrame(new_races_to_add)
    races_updated = pd.concat([races, new_races_df], ignore_index=True)
    races_updated.to_csv('data-raw/races.csv', index=False)
    print(f"   Added {len(new_races_to_add)} new races to races.csv")
