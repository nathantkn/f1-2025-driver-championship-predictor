#!/usr/bin/env python3
"""
Replace target_bucket column in base.csv with actual finishing positions from results.csv.
DNF/DSQ entries (position = backslash-N) are set to 20.
"""

import pandas as pd
import numpy as np

def main():
    print("Replacing target_bucket with finishing positions...")
    
    # Load data
    base = pd.read_csv('data-cleaned/base.csv')
    results = pd.read_csv('data-raw/results.csv')
    
    print(f"Base.csv before: {len(base)} rows")
    print(f"Columns: {base.columns.tolist()}")
    print()
    
    # Drop target_bucket and target_position if they exist
    if 'target_bucket' in base.columns:
        base = base.drop(columns=['target_bucket'])
        print("✓ Dropped target_bucket column")
    
    if 'target_position' in base.columns:
        base = base.drop(columns=['target_position'])
        print("✓ Dropped existing target_position column")
    
    # Prepare results data with positions
    results_positions = results[['raceId', 'driverId', 'constructorId', 'grid', 'position']].copy()
    
    # Convert position to numeric, treating '\N' as NaN
    results_positions['target_position'] = pd.to_numeric(results_positions['position'], errors='coerce')
    
    # Replace NaN (DNF/DSQ) with 20
    results_positions['target_position'] = results_positions['target_position'].fillna(20)
    
    # Convert to integer
    results_positions['target_position'] = results_positions['target_position'].astype(int)
    
    # Drop the original position column
    results_positions = results_positions.drop(columns=['position'])
    
    print(f"\nResults positions prepared: {len(results_positions)} rows")
    print(f"Position distribution:")
    print(results_positions['target_position'].value_counts().sort_index().head(25).to_string())
    print()
    
    # Merge with base
    base_with_position = base.merge(
        results_positions,
        on=['raceId', 'driverId', 'constructorId', 'grid'],
        how='left'
    )
    
    print(f"\nBase.csv after merge: {len(base_with_position)} rows")
    print(f"Non-null target_position: {base_with_position['target_position'].notna().sum()}")
    print(f"Null target_position: {base_with_position['target_position'].isna().sum()}")
    print()
    
    # Remove rows where target_position is null (no matching result)
    base_clean = base_with_position[base_with_position['target_position'].notna()].copy()
    
    print(f"After removing nulls: {len(base_clean)} rows")
    print()
    
    # Save updated base.csv
    base_clean.to_csv('data-cleaned/base.csv', index=False)
    
    print("✅ Saved updated base.csv")
    print(f"   Total rows: {len(base_clean)}")
    print(f"   Columns: {len(base_clean.columns)}")
    print(f"   Target position range: {base_clean['target_position'].min()}-{base_clean['target_position'].max()}")
    print()
    
    # Show sample
    print("Sample rows:")
    print(base_clean[['year', 'round', 'driverId', 'grid', 'target_position']].head(10).to_string())

if __name__ == '__main__':
    main()
