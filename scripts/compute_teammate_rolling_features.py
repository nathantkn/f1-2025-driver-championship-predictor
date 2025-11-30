import pandas as pd

# Load data
results = pd.read_csv('data-raw/results.csv')
races = pd.read_csv('data-raw/races.csv')
base = pd.read_csv('data-cleaned/base_with_driver_standings_with_features.csv')

# Merge to get date and year for each result
results = results.merge(
    races[['raceId', 'date', 'year']],
    on='raceId',
    how='left'
)
results['date'] = pd.to_datetime(results['date'])

# Only use races from 2010 onwards
results = results[results['year'] >= 2010].copy()

# Helper column for top10
results['is_top10'] = (results['points'] > 0).astype(int)

# Map race date to base

# Merge base with results to get the correct date for each (raceId, driverId)

# Use the existing 'date' column in base
base['constructorId'] = base['constructorId'].astype(int)

# Function to compute teammate rolling features for each row in base
def teammate_rolling(row):
    raceId = row['raceId']
    driverId = row['driverId']
    constructorId = row['constructorId']
    date = row['date']
    teammates = results[
        (results['constructorId'] == constructorId) &
        (results['driverId'] != driverId) &
        (results['date'] < date)
    ].sort_values('date')
    import numpy as np
    def ceil2(x):
        return np.ceil(x * 100) / 100 if pd.notnull(x) else x
    last3 = teammates.tail(3)
    tm_last3_avg_points = ceil2(last3['points'].mean()) if not last3.empty else None
    tm_last3_top10_rate = ceil2(last3['is_top10'].mean()) if not last3.empty else None
    return pd.Series([tm_last3_avg_points, tm_last3_top10_rate])

base[['tm_last3_avg_points', 'tm_last3_top10_rate']] = base.apply(teammate_rolling, axis=1)

base.to_csv('data-cleaned/base_with_driver_standings_with_features.csv', index=False)
print('Teammate rolling features merged and saved.')
