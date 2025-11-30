#!/usr/bin/env python3
"""
Integrate 2025 Sprint Race results into base.csv
- Maps sprint results to proper raceIds
- Inserts sprints chronologically before their GP
- Recalculates all rolling features in order
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*70)
print("INTEGRATING 2025 SPRINT RACES INTO BASE.CSV")
print("="*70)

# Load data
base = pd.read_csv('data-cleaned/base.csv')
sprint_results = pd.read_csv('data-raw/Formula1_2025Season_SprintResults.csv')
races = pd.read_csv('data-raw/races.csv')
drivers = pd.read_csv('data-raw/drivers.csv')
constructors = pd.read_csv('data-raw/constructors.csv')
results = pd.read_csv('data-raw/results.csv')

# Create driver name mapping
drivers['full_name'] = drivers['forename'] + ' ' + drivers['surname']
driver_name_to_id = dict(zip(drivers['full_name'], drivers['driverId']))

# Team name mapping (handling 2025 team names)
team_mapping = {
    'McLaren Mercedes': 'McLaren',
    'Red Bull Racing Honda RBPT': 'Red Bull',
    'Mercedes': 'Mercedes',
    'Ferrari': 'Ferrari',
    'Aston Martin Aramco Mercedes': 'Aston Martin',
    'Williams Mercedes': 'Williams',
    'Racing Bulls Honda RBPT': 'RB',
    'Alpine Renault': 'Alpine',
    'Haas Ferrari': 'Haas',
    'Kick Sauber Ferrari': 'Sauber'
}

constructors['name_clean'] = constructors['name'].str.strip()
constructor_mapping = {}
for team_full, team_short in team_mapping.items():
    matching = constructors[constructors['name_clean'].str.contains(team_short, case=False, na=False)]
    if len(matching) > 0:
        constructor_mapping[team_full] = matching.iloc[-1]['constructorId']  # Use most recent

print(f"\nOriginal base.csv: {len(base)} rows")
print(f"Sprint results to add: {len(sprint_results)} entries")

# Map sprint tracks to rounds and create sprint race entries
sprint_track_to_round = {
    'China': 2,
    'Miami': 6,
    'Belgium': 13,
    'United States': 19,
    'Brazil': 21
}

# Create sprint race IDs (use fractional round numbers for ordering)
sprint_race_data = []

for track_name, gp_round in sprint_track_to_round.items():
    track_sprints = sprint_results[sprint_results['Track'] == track_name].copy()
    
    # Get the GP race info
    gp_race = races[(races['year'] == 2025) & (races['round'] == gp_round)].iloc[0]
    circuitId = gp_race['circuitId']
    
    # Create sprint raceId (GP raceId - 0.5 for ordering purposes, will use actual GP raceId + 1000)
    sprint_raceId = gp_race['raceId'] + 1000
    
    # Get GP date and set sprint date to 1 day before
    gp_date = pd.to_datetime(gp_race['date'])
    sprint_date = gp_date - timedelta(days=1)
    
    for idx, row in track_sprints.iterrows():
        driver_name = row['Driver']
        team_name = row['Team']
        
        # Map driver and constructor
        driverId = driver_name_to_id.get(driver_name)
        constructorId = constructor_mapping.get(team_name)
        
        if driverId is None or constructorId is None:
            print(f"Warning: Could not map {driver_name} ({team_name})")
            continue
        
        # Handle position (DNF/NC -> 20)
        position = row['Position']
        if position in ['NC', 'DNF'] or pd.isna(position):
            target_position = 20
        else:
            try:
                target_position = int(position)
            except:
                target_position = 20
        
        sprint_race_data.append({
            'raceId': sprint_raceId,
            'driverId': driverId,
            'constructorId': constructorId,
            'grid': row['Starting Grid'],
            'year': 2025,
            'round': gp_round,  # Same round as GP
            'round_order': gp_round - 0.1,  # For sorting: sprint before GP
            'circuitId': circuitId,
            'is_sprint_weekend': 1,
            'date': sprint_date.strftime('%Y-%m-%d'),
            'target_position': target_position,
            'points_scored': row['Points']
        })

sprint_df = pd.DataFrame(sprint_race_data)
print(f"\nCreated {len(sprint_df)} sprint race entries")
print(f"Sprint races: {sprint_df.groupby('round')['raceId'].first().to_dict()}")

# Add round_order to base for sorting
base['round_order'] = base['round'].astype(float)

# Filter to pre-2025 data only
base_pre2025 = base[base['year'] < 2025].copy()

print(f"\nBase data before 2025: {len(base_pre2025)} rows")

# Now we need to rebuild 2025 data from scratch with sprints included
# Get all 2025 GP results
results_2025 = results.merge(races[['raceId', 'year', 'round', 'circuitId', 'date']], on='raceId')
results_2025 = results_2025[results_2025['year'] == 2025].copy()

# Add target_position from results
results_2025['target_position'] = pd.to_numeric(results_2025['position'], errors='coerce').fillna(20).astype(int)
results_2025['round_order'] = results_2025['round'].astype(float)
results_2025['is_sprint_weekend'] = 0  # GPs are not sprint races themselves

# Combine sprint and GP data for 2025
sprint_df_subset = sprint_df[['raceId', 'driverId', 'constructorId', 'grid', 'year', 'round', 
                               'round_order', 'circuitId', 'is_sprint_weekend', 'date', 'target_position']].copy()
sprint_df_subset['points_scored'] = sprint_df['points_scored']

gp_2025_subset = results_2025[['raceId', 'driverId', 'constructorId', 'grid', 'year', 'round', 
                                'round_order', 'circuitId', 'is_sprint_weekend', 'date', 'target_position']].copy()
gp_2025_subset['points_scored'] = results_2025['points']

# Combine and sort by round_order
combined_2025 = pd.concat([sprint_df_subset, gp_2025_subset], ignore_index=True)
combined_2025 = combined_2025.sort_values(['round_order', 'raceId', 'driverId']).reset_index(drop=True)

print(f"\nCombined 2025 data (sprint + GP): {len(combined_2025)} rows")
print(f"Rounds with data: {sorted(combined_2025['round'].unique())}")

# Now recalculate ALL rolling features for 2025 data in chronological order
print("\nRecalculating rolling features for 2025...")

# Initialize columns
combined_2025['prev_raceId'] = 0
combined_2025['cum_points_before'] = 0.0
combined_2025['position_before'] = 0
combined_2025['wins_before'] = 0
combined_2025['drv_last3_avg_points'] = 0.0
combined_2025['drv_last3_top10_rate'] = 0.0
combined_2025['drv_last3_dnf_rate'] = 0.0
combined_2025['drv_last3_avg_grid'] = 0.0
combined_2025['tm_last3_avg_points'] = 0.0
combined_2025['tm_last3_top10_rate'] = 0.0
combined_2025['drv_track_prev_starts'] = 0
combined_2025['drv_track_prev_avg_points'] = 0.0

# Get 2024 final standings for initial cum_points_before
results_2024 = results.merge(races[['raceId', 'year']], on='raceId')
results_2024 = results_2024[results_2024['year'] == 2024]
final_2024_points = results_2024.groupby('driverId')['points'].sum().to_dict()
final_2024_wins = results_2024[results_2024['position'] == '1'].groupby('driverId').size().to_dict()

# Create history tracker including all historical data + 2025 races processed so far
all_history = base_pre2025.copy()

# Process each unique race in order
unique_races = combined_2025[['round_order', 'raceId']].drop_duplicates().sort_values('round_order')

for _, race_info in unique_races.iterrows():
    round_order = race_info['round_order']
    raceId = race_info['raceId']
    
    race_entries = combined_2025[(combined_2025['round_order'] == round_order) & 
                                  (combined_2025['raceId'] == raceId)].copy()
    
    is_first_race = (round_order == combined_2025['round_order'].min())
    
    for idx in race_entries.index:
        row = combined_2025.loc[idx]
        driverId = row['driverId']
        constructorId = row['constructorId']
        circuitId = row['circuitId']
        
        # Get driver's history up to this point
        driver_history = all_history[all_history['driverId'] == driverId].copy()
        
        if is_first_race:
            # First race of 2025
            combined_2025.at[idx, 'cum_points_before'] = final_2024_points.get(driverId, 0)
            combined_2025.at[idx, 'wins_before'] = final_2024_wins.get(driverId, 0)
        else:
            # Get cumulative from previous 2025 races
            prev_2025 = all_history[(all_history['driverId'] == driverId) & 
                                     (all_history['year'] == 2025)]
            if len(prev_2025) > 0:
                combined_2025.at[idx, 'cum_points_before'] = prev_2025['points_scored'].sum()
                combined_2025.at[idx, 'wins_before'] = (prev_2025['target_position'] == 1).sum()
        
        # Last 3 races stats
        last_3 = driver_history.sort_values('date').tail(3)
        if len(last_3) > 0:
            if 'points_scored' in last_3.columns:
                combined_2025.at[idx, 'drv_last3_avg_points'] = last_3['points_scored'].mean()
                combined_2025.at[idx, 'drv_last3_top10_rate'] = (last_3['points_scored'] > 0).mean()
                combined_2025.at[idx, 'drv_last3_dnf_rate'] = (last_3['points_scored'] == 0).mean()
            combined_2025.at[idx, 'drv_last3_avg_grid'] = last_3['grid'].mean()
        
        # Teammate last 3
        teammate_history = all_history[(all_history['constructorId'] == constructorId) &
                                        (all_history['driverId'] != driverId)]
        teammate_last3 = teammate_history.sort_values('date').tail(3)
        if len(teammate_last3) > 0 and 'points_scored' in teammate_last3.columns:
            combined_2025.at[idx, 'tm_last3_avg_points'] = teammate_last3['points_scored'].mean()
            combined_2025.at[idx, 'tm_last3_top10_rate'] = (teammate_last3['points_scored'] > 0).mean()
        
        # Track history
        track_history = driver_history[driver_history['circuitId'] == circuitId]
        if len(track_history) > 0:
            combined_2025.at[idx, 'drv_track_prev_starts'] = len(track_history)
            if 'points_scored' in track_history.columns:
                combined_2025.at[idx, 'drv_track_prev_avg_points'] = track_history['points_scored'].mean()
    
    # Calculate championship positions after this race
    standings = combined_2025[combined_2025['round_order'] <= round_order].groupby('driverId').agg({
        'points_scored': 'sum'
    }).reset_index()
    standings = standings.sort_values('points_scored', ascending=False).reset_index(drop=True)
    standings['position'] = range(1, len(standings) + 1)
    position_map = dict(zip(standings['driverId'], standings['position']))
    
    # Update position_before for NEXT race
    next_races = combined_2025[combined_2025['round_order'] > round_order]
    for idx in next_races.index:
        driverId = combined_2025.at[idx, 'driverId']
        combined_2025.at[idx, 'position_before'] = position_map.get(driverId, 99)
    
    # Add this race to history
    race_for_history = race_entries.copy()
    all_history = pd.concat([all_history, race_for_history], ignore_index=True)

# Round numeric columns
float_cols = ['cum_points_before', 'drv_last3_avg_points', 'drv_last3_top10_rate', 
              'drv_last3_dnf_rate', 'drv_last3_avg_grid', 'tm_last3_avg_points', 
              'tm_last3_top10_rate', 'drv_track_prev_avg_points']
for col in float_cols:
    if col in combined_2025.columns:
        combined_2025[col] = combined_2025[col].round(2)

# Drop temporary columns
combined_2025 = combined_2025.drop(columns=['points_scored', 'round_order'])
base_pre2025 = base_pre2025.drop(columns=['round_order'])

# Combine pre-2025 and new 2025 data
final_base = pd.concat([base_pre2025, combined_2025], ignore_index=True)
final_base = final_base.sort_values(['year', 'round', 'raceId', 'driverId']).reset_index(drop=True)

print(f"\n✅ Final base.csv: {len(final_base)} rows")
print(f"   Pre-2025: {len(base_pre2025)}")
print(f"   2025: {len(combined_2025)}")
print(f"   Added sprints: {len(sprint_df)}")

# Save
final_base.to_csv('data-cleaned/base.csv', index=False)
print("\n✅ Saved updated base.csv with sprint races integrated")

# Show sample
print("\n2025 data sample (first few entries per round):")
sample_2025 = final_base[final_base['year'] == 2025].groupby(['round', 'is_sprint_weekend']).head(2)
print(sample_2025[['round', 'raceId', 'is_sprint_weekend', 'driverId', 'grid', 'target_position', 'cum_points_before']].to_string())
