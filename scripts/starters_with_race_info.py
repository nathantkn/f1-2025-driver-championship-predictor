import pandas as pd

# Load the data
results = pd.read_csv('data-raw/results.csv')
races = pd.read_csv('data-raw/races.csv')

# Filter starters (grid not null, including grid == 0)
starters = results[results['grid'].notnull()]

# Select unique (raceId, driverId, constructorId, grid)
unique_starters = starters.drop_duplicates(subset=['raceId', 'driverId', 'constructorId', 'grid'])

# Join to races.csv on raceId
merged = unique_starters.merge(
    races[['raceId', 'year', 'round', 'circuitId', 'is_sprint_weekend', 'date', 'prev_raceId']],
    on='raceId',
    how='left'
)

# Filter to only races from 2010 onwards
merged = merged[merged['year'] >= 2010]

# Save to CSV
merged[['raceId', 'driverId', 'constructorId', 'grid', 'year', 'round', 'circuitId', 'is_sprint_weekend', 'date', 'prev_raceId']].to_csv('starters_with_race_info.csv', index=False)

print('Saved to starters_with_race_info.csv (2010 and later)')
