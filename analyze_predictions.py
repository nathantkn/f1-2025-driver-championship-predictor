"""
Analyze what goes into K-NN predictions for championship contenders
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'models')

# Import the custom transformer so we can load the model
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Define the WeightedFeatureTransformer class (needed to unpickle the model)
class WeightedFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessor, recent_weight=3.0, season_weight=2.0, other_weight=0.3):
        self.preprocessor = preprocessor
        self.recent_weight = recent_weight
        self.season_weight = season_weight
        self.other_weight = other_weight
        
    def fit(self, X, y=None):
        self.preprocessor.fit(X, y)
        # Feature counts after preprocessing
        categorical_features = ['circuitId']
        passthrough_features = ['is_sprint_weekend']
        recent_form_features = ['grid', 'drv_last3_avg_points', 'drv_last3_avg_grid', 'drv_last3_top10_rate']
        season_standing_features = ['cum_points_before', 'position_before', 'wins_before']
        other_features = ['drv_last3_dnf_rate', 'tm_last3_avg_points', 'tm_last3_top10_rate', 
                         'drv_track_prev_starts', 'drv_track_prev_avg_points']
        
        self.cat_features_ = len(self.preprocessor.named_transformers_['cat'].get_feature_names_out())
        self.pass_features_ = len(passthrough_features)
        self.recent_features_ = len(recent_form_features)
        self.season_features_ = len(season_standing_features)
        self.other_features_ = len(other_features)
        return self
        
    def transform(self, X):
        X_transformed = self.preprocessor.transform(X)
        
        # Apply weights to different feature groups
        start_idx = 0
        
        # Categorical and passthrough features (no weighting)
        end_idx = self.cat_features_ + self.pass_features_
        start_idx = end_idx
        
        # Recent form features (higher weight)
        end_idx = start_idx + self.recent_features_
        X_transformed[:, start_idx:end_idx] *= self.recent_weight
        start_idx = end_idx
        
        # Season standing features (medium weight)
        end_idx = start_idx + self.season_features_
        X_transformed[:, start_idx:end_idx] *= self.season_weight
        start_idx = end_idx
        
        # Other features (lower weight)
        end_idx = start_idx + self.other_features_
        X_transformed[:, start_idx:end_idx] *= self.other_weight
        
        return X_transformed

import joblib

# Load model and data
print("Loading model and data...")
model = joblib.load('trained_knn_pipeline.joblib')
remaining = pd.read_csv('data-cleaned/remaining_races.csv')
drivers_df = pd.read_csv('data-raw/drivers.csv')
base = pd.read_csv('data-cleaned/base.csv')

# Point mapping
midpoints = {'0': 0, '1–3': 2, '4–6': 5, '7–10': 8.5, '12–15': 13.5, '18–26': 22}
point_ranges = {
    '0': '0 points (DNF/DNS)',
    '1–3': '1-3 points (8th-10th)',
    '4–6': '4-6 points (5th-7th)',
    '7–10': '7-10 points (4th-5th)',
    '12–15': '12-15 points (2nd-3rd)',
    '18–26': '18-26 points (1st-2nd)'
}

# Feature columns used by model
categorical_features = ['circuitId']
passthrough_features = ['is_sprint_weekend']
numeric_features = [
    'grid', 'cum_points_before', 'position_before', 'wins_before',
    'drv_last3_avg_points', 'drv_last3_top10_rate', 'drv_last3_dnf_rate', 'drv_last3_avg_grid',
    'tm_last3_avg_points', 'tm_last3_top10_rate',
    'drv_track_prev_starts', 'drv_track_prev_avg_points'
]

all_features = categorical_features + passthrough_features + numeric_features

# Analyze Round 23 (Qatar Sprint) for top 3 championship contenders
print("\n" + "="*100)
print("PREDICTION ANALYSIS - Round 23 (Qatar Sprint)")
print("="*100)

for driver_id, name in [(846, 'Lando Norris'), (857, 'Oscar Piastri'), (830, 'Max Verstappen')]:
    print(f"\n{'='*100}")
    print(f"{name.upper()} (driverId {driver_id})")
    print("="*100)
    
    # Get driver's Round 23 entry
    driver_entry = remaining[(remaining['driverId'] == driver_id) & (remaining['round'] == 23)].copy()
    
    if len(driver_entry) == 0:
        print(f"No entry found for {name}")
        continue
    
    # Prepare features
    for col in numeric_features:
        if col in driver_entry.columns:
            driver_entry[col] = driver_entry[col].fillna(0)
    
    X_driver = driver_entry[all_features]
    
    # Get prediction probabilities
    proba = model.predict_proba(X_driver)[0]
    predicted_class = model.predict(X_driver)[0]
    
    # Calculate expected points
    expected_pts = sum(proba[i] * midpoints[cls] for i, cls in enumerate(model.classes_))
    
    print("\nINPUT FEATURES:")
    print("-" * 100)
    print(f"  Grid Position: {driver_entry['grid'].values[0]:.2f}")
    print(f"  Championship Position: {int(driver_entry['position_before'].values[0])} ({driver_entry['cum_points_before'].values[0]:.0f} points)")
    print(f"  Wins This Season: {int(driver_entry['wins_before'].values[0])}")
    print(f"\n  Recent Form (Last 3 Races):")
    print(f"    - Avg Points: {driver_entry['drv_last3_avg_points'].values[0]:.2f}")
    print(f"    - Avg Grid: {driver_entry['drv_last3_avg_grid'].values[0]:.2f}")
    print(f"    - Top 10 Rate: {driver_entry['drv_last3_top10_rate'].values[0]:.2f}")
    print(f"    - DNF Rate: {driver_entry['drv_last3_dnf_rate'].values[0]:.2f}")
    print(f"\n  Teammate Form (Last 3 Races):")
    print(f"    - Avg Points: {driver_entry['tm_last3_avg_points'].values[0]:.2f}")
    print(f"    - Top 10 Rate: {driver_entry['tm_last3_top10_rate'].values[0]:.2f}")
    print(f"\n  Track History (Qatar):")
    print(f"    - Previous Starts: {int(driver_entry['drv_track_prev_starts'].values[0])}")
    print(f"    - Avg Points: {driver_entry['drv_track_prev_avg_points'].values[0]:.2f}")
    
    print(f"\nPREDICTION PROBABILITIES:")
    print("-" * 100)
    for i, cls in enumerate(model.classes_):
        prob_pct = proba[i] * 100
        bar = "█" * int(prob_pct / 2)
        print(f"  {cls:>6} ({point_ranges[cls]:30s}): {prob_pct:5.1f}% {bar}")
    
    print(f"\nFINAL PREDICTION:")
    print("-" * 100)
    print(f"  Most Likely Result: {predicted_class} ({point_ranges[predicted_class]})")
    print(f"  Expected Points (weighted average): {expected_pts:.2f}")
    
    # Show similar historical races from training data
    print(f"\nFINDING SIMILAR HISTORICAL RACES (K-NN neighbors)...")
    print("-" * 100)
    
    # Get the K-NN neighbors
    n_neighbors = model.named_steps['classifier'].n_neighbors
    
    # Transform the features
    X_transformed = model.named_steps['preprocessor'].transform(X_driver)
    
    # Get nearest neighbors
    distances, indices = model.named_steps['classifier'].kneighbors(X_transformed, n_neighbors=min(10, n_neighbors))
    
    print(f"\nTop 10 most similar historical races:")
    print(f"{'Race':<25} {'Grid':<6} {'Result':<8} {'Points':<7} {'Distance':<10}")
    print("-" * 100)
    
    # Get training data
    train_df = pd.read_parquet('driver_race_prerace_train.parquet')
    
    for i, (dist, idx) in enumerate(zip(distances[0][:10], indices[0][:10])):
        similar_race = train_df.iloc[idx]
        race_name = f"R{int(similar_race['round'])} {int(similar_race['year'])}"
        grid_pos = similar_race['grid']
        result_bucket = similar_race['target_bucket']
        points = midpoints[str(result_bucket)]
        
        print(f"{race_name:<25} {grid_pos:<6.1f} {result_bucket!s:<8} {points:<7.1f} {dist:<10.4f}")

print("\n" + "="*100)
print("SUMMARY")
print("="*100)
print("\nThe K-NN model predicts points by:")
print("1. Finding the 'k' most similar historical races (based on all input features)")
print("2. Weighting recent form 3x, championship position 2x, track history 0.3x")
print("3. Looking at what happened in those similar past races")
print("4. Taking a weighted average based on distance (closer races = more influence)")
print("\nLow predictions suggest the model finds many similar races where drivers scored fewer points,")
print("possibly due to incidents, bad luck, or underperformance - which are part of racing history.")
