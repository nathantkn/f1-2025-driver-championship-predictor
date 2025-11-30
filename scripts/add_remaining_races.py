"""
Add remaining 2025 races and generate predictions
- Round 23: Qatar Grand Prix Sprint (circuitId 78)
- Round 24: Qatar Grand Prix (circuitId 78)
- Round 25: Abu Dhabi Grand Prix (circuitId 24)
"""

import pandas as pd
import numpy as np
import joblib

# Load existing data
base = pd.read_csv('data-cleaned/base.csv')
races = pd.read_csv('data-raw/races.csv')
drivers = pd.read_csv('data-raw/drivers.csv')
results = pd.read_csv('data-raw/results.csv')

print("Adding 3 remaining 2025 races...")

# Add new races to races.csv
new_races = [
    {
        'raceId': 1167,
        'prev_raceId': 1166,
        'is_sprint_weekend': 1,
        'year': 2025,
        'round': 23,
        'circuitId': 78,
        'name': 'Qatar Grand Prix Sprint',
        'date': '2025-11-29',
        'time': '17:00:00',
        'url': '',
        'fp1_date': '2025-11-28',
        'fp1_time': '13:30:00',
        'fp2_date': '',
        'fp2_time': '',
        'fp3_date': '',
        'fp3_time': '',
        'quali_date': '2025-11-28',
        'quali_time': '17:00:00',
        'sprint_date': '2025-11-29',
        'sprint_time': '13:00:00'
    },
    {
        'raceId': 1168,
        'prev_raceId': 1167,
        'is_sprint_weekend': 1,
        'year': 2025,
        'round': 24,
        'circuitId': 78,
        'name': 'Qatar Grand Prix',
        'date': '2025-11-30',
        'time': '17:00:00',
        'url': '',
        'fp1_date': '2025-11-28',
        'fp1_time': '13:30:00',
        'fp2_date': '',
        'fp2_time': '',
        'fp3_date': '',
        'fp3_time': '',
        'quali_date': '2025-11-29',
        'quali_time': '17:00:00',
        'sprint_date': '',
        'sprint_time': ''
    },
    {
        'raceId': 1169,
        'prev_raceId': 1168,
        'is_sprint_weekend': 0,
        'year': 2025,
        'round': 25,
        'circuitId': 24,
        'name': 'Abu Dhabi Grand Prix',
        'date': '2025-12-08',
        'time': '17:00:00',
        'url': '',
        'fp1_date': '2025-12-06',
        'fp1_time': '13:30:00',
        'fp2_date': '2025-12-06',
        'fp2_time': '17:00:00',
        'fp3_date': '2025-12-07',
        'fp3_time': '14:30:00',
        'quali_date': '2025-12-07',
        'quali_time': '18:00:00',
        'sprint_date': '',
        'sprint_time': ''
    }
]

new_races_df = pd.DataFrame(new_races)
races_updated = pd.concat([races, new_races_df], ignore_index=True)
races_updated.to_csv('data-raw/races.csv', index=False)
print(f"Added 3 races to races.csv (total now: {len(races_updated)} races)")

# Get active drivers from round 22
last_round_drivers = base[base['round'] == 22][['driverId', 'constructorId']].drop_duplicates()
print(f"Found {len(last_round_drivers)} active drivers")

# Get latest standings after round 22
latest_standings = base[base['round'] == 22][['driverId', 'cum_points_before']].copy()
# Need to add points from round 22 to get standings before round 23
results_r22 = results.merge(races[['raceId', 'year', 'round']], on='raceId')
results_r22 = results_r22[(results_r22['year'] == 2025) & (results_r22['round'] == 22)]
points_r22 = results_r22.groupby('driverId')['points'].sum().reset_index()
points_r22.columns = ['driverId', 'points_r22']

latest_standings = latest_standings.merge(points_r22, on='driverId', how='left')
latest_standings['points_r22'] = latest_standings['points_r22'].fillna(0)
latest_standings['cum_points_before_r23'] = latest_standings['cum_points_before'] + latest_standings['points_r22']

# Calculate positions and wins
driver_stats = base[base['round'] == 22][['driverId']].drop_duplicates()
driver_stats = driver_stats.merge(latest_standings[['driverId', 'cum_points_before_r23']], on='driverId', how='left')
driver_stats['cum_points_before_r23'] = driver_stats['cum_points_before_r23'].fillna(0)
driver_stats = driver_stats.sort_values('cum_points_before_r23', ascending=False).reset_index(drop=True)
driver_stats['position_before'] = range(1, len(driver_stats) + 1)

# Count wins (need to check results for position=1)
wins = results.merge(races[['raceId', 'year', 'round']], on='raceId')
wins = wins[(wins['year'] == 2025) & (wins['round'] <= 22) & (wins['position'] == 1)]
wins_count = wins.groupby('driverId').size().reset_index(name='wins_before')
driver_stats = driver_stats.merge(wins_count, on='driverId', how='left')
driver_stats['wins_before'] = driver_stats['wins_before'].fillna(0)

# Prepare to calculate rolling features for rounds 23-25
# We need to recompute rolling features using data through round 22
results_with_dates = results.merge(races[['raceId', 'date', 'year', 'circuitId']], on='raceId', how='left')
results_with_dates['date'] = pd.to_datetime(results_with_dates['date'])
results_2010_on = results_with_dates[results_with_dates['year'] >= 2010].copy()
results_2010_on = results_2010_on.sort_values(['driverId', 'date'])

# Helper columns
results_2010_on['is_top10'] = (results_2010_on['points'] > 0).astype(int)
dnf_status_ids = set([3, 4, 5, 11, 12, 20])
results_2010_on['is_dnf'] = results_2010_on['statusId'].isin(dnf_status_ids).astype(int)

# Calculate rolling features for each driver
def rolling_features_updated(df):
    import numpy as np
    def ceil2(x):
        return np.ceil(x * 100) / 100 if pd.notnull(x) else x
    df = df.copy()
    df['drv_last3_avg_points'] = df['points'].shift(1).rolling(3, min_periods=1).mean().apply(ceil2)
    df['drv_last3_top10_rate'] = df['is_top10'].shift(1).rolling(3, min_periods=1).mean().apply(ceil2)
    df['drv_last3_dnf_rate'] = df['is_dnf'].shift(1).rolling(3, min_periods=1).mean().apply(ceil2)
    df['drv_last3_avg_grid'] = df['grid'].shift(1).rolling(3, min_periods=1).mean().apply(ceil2)
    return df

results_2010_on = results_2010_on.groupby('driverId', group_keys=False).apply(rolling_features_updated)

# Get track familiarity (previous starts and avg points at each circuit)
track_stats = results_2010_on.groupby(['driverId', 'circuitId']).agg({
    'resultId': 'count',
    'points': 'mean'
}).reset_index()
track_stats.columns = ['driverId', 'circuitId', 'drv_track_prev_starts', 'drv_track_prev_avg_points']

# Generate grid positions for future races (use recent avg as placeholder)
recent_grids = base[base['round'] >= 20].groupby('driverId')['grid'].mean().reset_index()
recent_grids.columns = ['driverId', 'avg_grid']

# Create entries for rounds 23, 24, 25
new_entries = []

for race_info in new_races:
    raceId = race_info['raceId']
    round_num = race_info['round']
    circuitId = race_info['circuitId']
    is_sprint = race_info['is_sprint_weekend']
    date = race_info['date']
    prev_raceId = race_info['prev_raceId']
    
    for idx, driver_row in last_round_drivers.iterrows():
        driverId = driver_row['driverId']
        constructorId = driver_row['constructorId']
        
        # Get standings before this race
        standings_row = driver_stats[driver_stats['driverId'] == driverId]
        if round_num == 23:
            cum_points = standings_row['cum_points_before_r23'].values[0] if len(standings_row) > 0 else 0
        else:
            # For rounds 24-25, we'll update after predicting round 23
            cum_points = standings_row['cum_points_before_r23'].values[0] if len(standings_row) > 0 else 0
        
        position = standings_row['position_before'].values[0] if len(standings_row) > 0 else None
        wins = standings_row['wins_before'].values[0] if len(standings_row) > 0 else 0
        
        # Get grid position (use average for now)
        grid_row = recent_grids[recent_grids['driverId'] == driverId]
        grid = int(np.ceil(grid_row['avg_grid'].values[0])) if len(grid_row) > 0 else 10
        
        # Get rolling features (last values from results_2010_on)
        driver_results = results_2010_on[results_2010_on['driverId'] == driverId].sort_values('date')
        if len(driver_results) > 0:
            last_result = driver_results.iloc[-1]
            drv_last3_avg_points = last_result['drv_last3_avg_points']
            drv_last3_top10_rate = last_result['drv_last3_top10_rate']
            drv_last3_dnf_rate = last_result['drv_last3_dnf_rate']
            drv_last3_avg_grid = last_result['drv_last3_avg_grid']
        else:
            drv_last3_avg_points = drv_last3_top10_rate = drv_last3_dnf_rate = drv_last3_avg_grid = None
        
        # Get teammate rolling features (recent teammate performance)
        teammate_results = results_2010_on[
            (results_2010_on['constructorId'] == constructorId) & 
            (results_2010_on['driverId'] != driverId)
        ].sort_values('date').tail(3)
        
        if len(teammate_results) > 0:
            tm_last3_avg_points = np.ceil(teammate_results['points'].mean() * 100) / 100
            tm_last3_top10_rate = np.ceil(teammate_results['is_top10'].mean() * 100) / 100
        else:
            tm_last3_avg_points = tm_last3_top10_rate = None
        
        # Get track familiarity
        track_row = track_stats[(track_stats['driverId'] == driverId) & (track_stats['circuitId'] == circuitId)]
        drv_track_prev_starts = track_row['drv_track_prev_starts'].values[0] if len(track_row) > 0 else 0
        drv_track_prev_avg_points = track_row['drv_track_prev_avg_points'].values[0] if len(track_row) > 0 else 0.0
        
        new_entries.append({
            'raceId': raceId,
            'driverId': driverId,
            'constructorId': constructorId,
            'grid': grid,
            'year': 2025,
            'round': round_num,
            'circuitId': circuitId,
            'is_sprint_weekend': is_sprint,
            'date': date,
            'prev_raceId': prev_raceId,
            'cum_points_before': cum_points,
            'position_before': position,
            'wins_before': wins,
            'drv_last3_avg_points': drv_last3_avg_points,
            'drv_last3_top10_rate': drv_last3_top10_rate,
            'drv_last3_dnf_rate': drv_last3_dnf_rate,
            'drv_last3_avg_grid': drv_last3_avg_grid,
            'tm_last3_avg_points': tm_last3_avg_points,
            'tm_last3_top10_rate': tm_last3_top10_rate,
            'drv_track_prev_starts': drv_track_prev_starts,
            'drv_track_prev_avg_points': drv_track_prev_avg_points
            # NO target_bucket - this is a future race
        })

new_entries_df = pd.DataFrame(new_entries)
print(f"Created {len(new_entries_df)} new race entries for rounds 23-25")

# Append to base.csv
base_updated = pd.concat([base, new_entries_df], ignore_index=True)
base_updated.to_csv('data-cleaned/base.csv', index=False)
print(f"Updated base.csv (total now: {len(base_updated)} rows)")

print("\n✅ Successfully added 3 remaining races!")
print("You can now re-run the f1_knn_pipeline.py script to generate predictions.")
