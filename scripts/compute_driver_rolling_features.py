import pandas as pd

# Load data
results = pd.read_csv('data-raw/results.csv')
races = pd.read_csv('data-raw/races.csv')

# Merge to get date and year for each result
results = results.merge(
    races[['raceId', 'date', 'year']],
    on='raceId',
    how='left'
)

# Only use races from 2010 onwards for rolling features
results = results[results['year'] >= 2010].copy()

# Convert date to datetime
results['date'] = pd.to_datetime(results['date'])

# Sort by driverId and date
results = results.sort_values(['driverId', 'date'])

# Define DNF/DNS statusId values (adjust as needed for your dataset)
dnf_status_ids = set([3, 4, 5, 11, 12, 20])  # Example: R, D, etc.

# Create helper columns
results['is_top10'] = (results['points'] > 0).astype(int)
results['is_dnf'] = results['statusId'].isin(dnf_status_ids).astype(int)

# Group by driverId and compute rolling features, shifted by 1 to exclude current race
def rolling_features(df):
    df = df.copy()
    import numpy as np
    def ceil2(x):
        return np.ceil(x * 100) / 100 if pd.notnull(x) else x
    df['drv_last3_avg_points'] = df['points'].shift(1).rolling(3, min_periods=1).mean().apply(ceil2)
    df['drv_last3_top10_rate'] = df['is_top10'].shift(1).rolling(3, min_periods=1).mean().apply(ceil2)
    df['drv_last3_dnf_rate'] = df['is_dnf'].shift(1).rolling(3, min_periods=1).mean().apply(ceil2)
    df['drv_last3_avg_grid'] = df['grid'].shift(1).rolling(3, min_periods=1).mean().apply(ceil2)
    return df
    return df

results = results.groupby('driverId', group_keys=False).apply(rolling_features)

# Select relevant columns for merge
features = results[['raceId', 'driverId', 'drv_last3_avg_points', 'drv_last3_top10_rate', 'drv_last3_dnf_rate', 'drv_last3_avg_grid']]

# Load your base file
base = pd.read_csv('data-cleaned/base_with_driver_standings.csv')

# Merge features back to base
base = base.merge(features, on=['raceId', 'driverId'], how='left')

# Save or display
base.to_csv('data-cleaned/base_with_driver_standings_with_features.csv', index=False)
print('Features merged and saved to base_with_driver_standings_with_features.csv')
