"""
F1 Championship Prediction Pipeline
Steps 7-10: Train/Validation Split, K-NN Pipeline, Tuning, and Championship Projection
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import log_loss, f1_score, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt

# ========================
# Step 7: Load Data & Train/Validation Split
# ========================

print("Step 7: Loading data and creating train/validation split...")
df = pd.read_csv('data-cleaned/base.csv')

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Define the target column and classes
target_col = 'target_bucket'
target_classes = ['0', '1–3', '4–6', '7–10', '12–15', '18–26']

# Ensure target is categorical with proper ordering
df[target_col] = pd.Categorical(df[target_col], categories=target_classes, ordered=True)

# Training set: all rows with targets (completed races)
train_df = df[df[target_col].notna()].copy()

# Load remaining races from dedicated CSV
import os
if os.path.exists('data-cleaned/remaining_races.csv'):
    holdout_df = pd.read_csv('data-cleaned/remaining_races.csv')
    holdout_df['date'] = pd.to_datetime(holdout_df['date'])
    latest_2025_round = train_df[train_df['year'] == 2025]['round'].max() if len(train_df[train_df['year'] == 2025]) > 0 else 0
    print(f"Latest completed 2025 round: {latest_2025_round}")
    print(f"Training set size: {len(train_df)} rows")
    print(f"Holdout set size: {len(holdout_df)} rows (from remaining_races.csv)")
else:
    print("No remaining_races.csv found - no races to predict")
    holdout_df = pd.DataFrame()
    latest_2025_round = train_df[train_df['year'] == 2025]['round'].max() if len(train_df[train_df['year'] == 2025]) > 0 else 0
    print(f"Latest completed 2025 round: {latest_2025_round}")
    print(f"Training set size: {len(train_df)} rows")
    print(f"Holdout set size: 0 rows")

# Remove rows with missing target
train_df = train_df[train_df[target_col].notna()].copy()
print(f"Training set after removing missing targets: {len(train_df)} rows")

# ========================
# Step 8: Build K-NN Pipeline
# ========================

print("\nStep 8: Building K-NN pipeline...")

# Define feature columns with importance groupings
categorical_features = ['circuitId']
passthrough_features = ['is_sprint_weekend']

# High importance: Recent form and current grid
recent_form_features = [
    'grid',
    'drv_last3_avg_points',
    'drv_last3_avg_grid',
    'drv_last3_top10_rate',
]

# Medium importance: Season standing
season_standing_features = [
    'cum_points_before',
    'position_before',
    'wins_before',
]

# Lower importance: Other features
other_features = [
    'drv_last3_dnf_rate',
    'tm_last3_avg_points',
    'tm_last3_top10_rate',
    'drv_track_prev_starts',
    'drv_track_prev_avg_points'
]

# Combine all numeric features
numeric_features = recent_form_features + season_standing_features + other_features

# Fill NaNs in numeric features with 0 (for early-career drivers)
for col in numeric_features:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(0)

# Create weighted preprocessor with different scalers for different feature groups
from sklearn.preprocessing import MinMaxScaler

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('pass', 'passthrough', passthrough_features),
        # Apply stronger scaling to recent form (weight = 3x)
        ('recent', StandardScaler(), recent_form_features),
        # Medium scaling for season standing (weight = 2x)
        ('season', StandardScaler(), season_standing_features),
        # Lower scaling for other features (weight = 0.1x for minimal track history influence)
        ('other', StandardScaler(), other_features)
    ],
    remainder='drop'
)

# Custom feature weighting wrapper
class WeightedFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessor, recent_weight=3.0, season_weight=2.0, other_weight=0.1):
        self.preprocessor = preprocessor
        self.recent_weight = recent_weight
        self.season_weight = season_weight
        self.other_weight = other_weight
        
    def fit(self, X, y=None):
        self.preprocessor.fit(X, y)
        # Calculate feature counts after preprocessing
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

# Create weighted preprocessor
weighted_preprocessor = WeightedFeatureTransformer(
    preprocessor,
    recent_weight=3.0,    # 3x weight for recent form (grid, points, top10 rate)
    season_weight=2.0,    # 2x weight for season standing (position, points, wins)
    other_weight=0.1      # 0.1x weight for track history and DNF rate (minimal influence)
)

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', weighted_preprocessor),
    ('classifier', KNeighborsClassifier())
])

# Define parameter grid - use 'distance' weighting to prioritize closer neighbors
param_grid = {
    'classifier__n_neighbors': [11, 21, 31, 41],
    'classifier__weights': ['distance'],  # Distance weighting gives more weight to similar historical races
    'classifier__metric': ['euclidean', 'manhattan']
}

# Custom scorer for expected points RMSE
def expected_points_rmse(y_true, y_pred_proba):
    """Calculate RMSE between actual and expected points"""
    # Midpoints for each bucket
    midpoints = {'0': 0, '1–3': 2, '4–6': 5, '7–10': 8.5, '12–15': 13.5, '18–26': 22}
    
    # Convert true labels to points
    true_points = np.array([midpoints[str(y)] for y in y_true])
    
    # Calculate expected points from probabilities
    expected_points = np.zeros(len(y_pred_proba))
    for i, class_name in enumerate(target_classes):
        expected_points += y_pred_proba[:, i] * midpoints[class_name]
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((true_points - expected_points) ** 2))
    return -rmse  # Negative because sklearn maximizes scores

# Prepare features and target
X_train = train_df[categorical_features + passthrough_features + numeric_features]
y_train = train_df[target_col].astype(str)

print(f"Feature matrix shape: {X_train.shape}")
print(f"Target distribution:\n{y_train.value_counts()}")

# ========================
# Step 8b: Hyperparameter Tuning with CV
# ========================

print("\nPerforming hyperparameter tuning with 5-fold Stratified CV...")

# Use log loss as the primary scoring metric
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='neg_log_loss',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV log-loss: {-grid_search.best_score_:.4f}")

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate on training set
y_train_pred = best_model.predict(X_train)
y_train_proba = best_model.predict_proba(X_train)

train_logloss = log_loss(y_train, y_train_proba)
train_f1 = f1_score(y_train, y_train_pred, average='macro')

print(f"\nTraining metrics:")
print(f"  Log-loss: {train_logloss:.4f}")
print(f"  Macro F1: {train_f1:.4f}")

# ========================
# Step 8c: Walk-Forward LOOCV
# ========================

print("\nPerforming walk-forward LOOCV by season/round...")

# Sort by year and round
train_df_sorted = train_df.sort_values(['year', 'round']).reset_index(drop=True)

# Get unique (year, round) combinations
unique_races = train_df_sorted[['year', 'round']].drop_duplicates().sort_values(['year', 'round'])

loocv_predictions = []
loocv_actuals = []

for idx, (year, round_num) in enumerate(unique_races.values):
    if idx == 0:
        continue  # Skip first race as we need prior data
    
    # Train on all races before this one
    train_mask_loocv = (train_df_sorted['year'] < year) | \
                       ((train_df_sorted['year'] == year) & (train_df_sorted['round'] < round_num))
    test_mask_loocv = (train_df_sorted['year'] == year) & (train_df_sorted['round'] == round_num)
    
    X_train_loocv = train_df_sorted.loc[train_mask_loocv, categorical_features + passthrough_features + numeric_features]
    y_train_loocv = train_df_sorted.loc[train_mask_loocv, target_col].astype(str)
    X_test_loocv = train_df_sorted.loc[test_mask_loocv, categorical_features + passthrough_features + numeric_features]
    y_test_loocv = train_df_sorted.loc[test_mask_loocv, target_col].astype(str)
    
    if len(y_train_loocv) < 50:  # Skip if too little training data
        continue
    
    # Clone and fit the best model
    temp_model = grid_search.best_estimator_
    temp_model.fit(X_train_loocv, y_train_loocv)
    
    # Predict
    y_pred_proba_loocv = temp_model.predict_proba(X_test_loocv)
    
    loocv_predictions.extend(y_pred_proba_loocv)
    loocv_actuals.extend(y_test_loocv)

if loocv_predictions:
    loocv_logloss = log_loss(loocv_actuals, np.array(loocv_predictions))
    print(f"Walk-forward LOOCV log-loss: {loocv_logloss:.4f}")

# ========================
# Step 9: Refit and Project Championship
# ========================

print("\nStep 9: Refitting on full training set and projecting 2025 championship...")

# Refit on full training set
best_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(best_model, 'trained_knn_pipeline.joblib')
print("Saved trained model to: trained_knn_pipeline.joblib")

# Load driver names for display
drivers = pd.read_csv('data-raw/drivers.csv')

# Get current championship standings (up to latest completed round)
if latest_2025_round > 0:
    current_standings = train_df[train_df['year'] == 2025].groupby('driverId').agg({
        'cum_points_before': 'max'
    }).reset_index()
    current_standings.columns = ['driverId', 'current_points']
    
    # Get the actual points scored in the completed 2025 races
    # We need to read the results to get actual points
    results = pd.read_csv('data-raw/results.csv')
    races = pd.read_csv('data-raw/races.csv')
    
    results_2025 = results.merge(races[['raceId', 'year', 'round']], on='raceId')
    results_2025 = results_2025[(results_2025['year'] == 2025) & (results_2025['round'] <= latest_2025_round)]
    
    actual_points_2025 = results_2025.groupby('driverId')['points'].sum().reset_index()
    actual_points_2025.columns = ['driverId', 'points_scored_2025']
    
    current_standings = current_standings.merge(actual_points_2025, on='driverId', how='left')
    current_standings['points_scored_2025'] = current_standings['points_scored_2025'].fillna(0)
    current_standings['total_points'] = current_standings['points_scored_2025']
else:
    # No 2025 races completed yet, start from 0
    current_standings = train_df.groupby('driverId').agg({
        'driverId': 'first'
    }).reset_index(drop=True)
    current_standings['total_points'] = 0

# Add driver names
current_standings = current_standings.merge(
    drivers[['driverId', 'forename', 'surname']], 
    on='driverId', 
    how='left'
)
current_standings['driver_name'] = current_standings['forename'] + ' ' + current_standings['surname']

print(f"Current standings (top 10):")
print(current_standings.nlargest(10, 'total_points')[['driver_name', 'total_points']].to_string(index=False))

# Project remaining races SEQUENTIALLY (updating rolling features after each race)
if len(holdout_df) > 0:
    print(f"\nProjecting {len(holdout_df)} remaining race entries sequentially...")
    
    # Point mapping for expected values
    midpoints = {'0': 0, '1–3': 2, '4–6': 5, '7–10': 8.5, '12–15': 13.5, '18–26': 22}
    
    # Get unique rounds to predict (sorted)
    remaining_rounds = sorted(holdout_df['round'].unique())
    print(f"Predicting rounds: {remaining_rounds}")
    
    # Create a combined dataset for rolling calculations (training + predicted)
    # Add actual points to training data for rolling calculations
    combined_df = train_df.copy()
    # Convert categorical target_bucket to numeric points (use midpoints)
    combined_df['predicted_points'] = combined_df['target_bucket'].astype(str).map(midpoints)
    
    # Track predicted results for each driver-race
    all_predictions = []
    
    # Predict each round sequentially
    for round_num in remaining_rounds:
        print(f"\n  Predicting Round {round_num}...")
        
        # Get entries for this round
        round_df = holdout_df[holdout_df['round'] == round_num].copy()
        
        # Update rolling features based on combined_df (which includes previous predictions)
        for idx, row in round_df.iterrows():
            driverId = row['driverId']
            constructorId = row['constructorId']
            
            # Get last 3 races for this driver from combined_df
            driver_history = combined_df[
                (combined_df['driverId'] == driverId) & 
                (combined_df['round'] < round_num)
            ].sort_values('round').tail(3)
            
            # Recalculate driver rolling features
            if len(driver_history) > 0:
                round_df.at[idx, 'drv_last3_avg_points'] = np.ceil(driver_history['predicted_points'].mean() * 100) / 100
                round_df.at[idx, 'drv_last3_top10_rate'] = np.ceil((driver_history['predicted_points'] > 0).mean() * 100) / 100
                # DNF rate: count races with 0 predicted points (representing DNF)
                round_df.at[idx, 'drv_last3_dnf_rate'] = np.ceil((driver_history['predicted_points'] == 0).mean() * 100) / 100
                round_df.at[idx, 'drv_last3_avg_grid'] = np.ceil(driver_history['grid'].mean() * 100) / 100
            
            # Get last 3 races for teammates
            teammate_history = combined_df[
                (combined_df['constructorId'] == constructorId) &
                (combined_df['driverId'] != driverId) &
                (combined_df['round'] < round_num)
            ].sort_values('round').tail(3)
            
            # Recalculate teammate rolling features
            if len(teammate_history) > 0:
                round_df.at[idx, 'tm_last3_avg_points'] = np.ceil(teammate_history['predicted_points'].mean() * 100) / 100
                round_df.at[idx, 'tm_last3_top10_rate'] = np.ceil((teammate_history['predicted_points'] > 0).mean() * 100) / 100
        
        # Prepare features for prediction
        for col in numeric_features:
            if col in round_df.columns:
                round_df[col] = round_df[col].fillna(0)
        
        X_round = round_df[categorical_features + passthrough_features + numeric_features]
        
        # Predict probabilities
        y_round_proba = best_model.predict_proba(X_round)
        
        # Convert to expected points
        expected_points = np.zeros(len(y_round_proba))
        for i, class_name in enumerate(target_classes):
            expected_points += y_round_proba[:, i] * midpoints[class_name]
        
        round_df['predicted_points'] = expected_points
        
        # Store predictions for final aggregation
        all_predictions.append(round_df[['driverId', 'round', 'predicted_points']])
        
        # Add predictions to combined_df for next round's rolling calculations
        # Only add the essential columns needed for rolling calculations
        round_df_for_rolling = round_df[['raceId', 'driverId', 'constructorId', 'grid', 'year', 'round', 
                                           'drv_last3_avg_points', 'drv_last3_top10_rate', 'drv_last3_dnf_rate',
                                           'drv_last3_avg_grid', 'tm_last3_avg_points', 'tm_last3_top10_rate',
                                           'predicted_points']].copy()
        combined_df = pd.concat([combined_df, round_df_for_rolling], ignore_index=True)
        
        print(f"    Predicted {len(round_df)} entries for Round {round_num}")
    
    # Combine all predictions with their updated rolling features
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    # Save the updated remaining_races.csv with sequential rolling features
    # Merge the predictions back with the original holdout_df to get all original columns
    # plus the updated rolling features from the sequential prediction
    
    # For each round, get the updated rolling features
    updated_rounds = []
    for round_num in remaining_rounds:
        # Get original holdout data for this round
        original_round = holdout_df[holdout_df['round'] == round_num].copy()
        
        # Get the predictions for this round (which have updated features)
        pred_round = all_predictions_df[all_predictions_df['round'] == round_num].copy()
        
        # Update rolling features in original_round based on what was calculated
        # Find the corresponding entries in combined_df (which has the updated features)
        for idx, row in original_round.iterrows():
            # Get the updated features from combined_df (filter for 2025 and this specific driver/round)
            updated_row = combined_df[
                (combined_df['year'] == 2025) &
                (combined_df['driverId'] == row['driverId']) & 
                (combined_df['round'] == round_num)
            ].tail(1)  # Get the last (most recent) entry for this driver-round
            
            if len(updated_row) > 0:
                # Update rolling features
                for col in ['drv_last3_avg_points', 'drv_last3_top10_rate', 'drv_last3_dnf_rate',
                           'drv_last3_avg_grid', 'tm_last3_avg_points', 'tm_last3_top10_rate']:
                    if col in updated_row.columns:
                        original_round.at[idx, col] = updated_row[col].values[0]
            
            # Update cum_points_before for subsequent rounds
            # For rounds after 23, add the predicted points from previous rounds
            if round_num > min(remaining_rounds):
                # Get predicted points from all previous rounds
                prev_rounds_points = all_predictions_df[
                    (all_predictions_df['driverId'] == row['driverId']) &
                    (all_predictions_df['round'] < round_num)
                ]['predicted_points'].sum()
                
                # Add to the original cum_points_before (which is points after round 22)
                original_round.at[idx, 'cum_points_before'] = row['cum_points_before'] + prev_rounds_points
        
        updated_rounds.append(original_round)
    
    updated_remaining_export = pd.concat(updated_rounds, ignore_index=True)
    
    # Round the rolling features to 2 decimals for readability
    float_cols = ['drv_last3_avg_points', 'drv_last3_top10_rate', 'drv_last3_dnf_rate',
                  'drv_last3_avg_grid', 'tm_last3_avg_points', 'tm_last3_top10_rate',
                  'drv_track_prev_avg_points']
    for col in float_cols:
        if col in updated_remaining_export.columns:
            updated_remaining_export[col] = updated_remaining_export[col].round(2)
    
    # Sort by round and driverId for consistency
    updated_remaining_export = updated_remaining_export.sort_values(['round', 'driverId']).reset_index(drop=True)
    
    # Save updated remaining_races.csv
    updated_remaining_export.to_csv('data-cleaned/remaining_races.csv', index=False)
    print(f"\n  ✅ Updated remaining_races.csv with {len(updated_remaining_export)} entries (sequential rolling features)")
    
    # Aggregate expected points by driver
    projected_points = all_predictions_df.groupby('driverId')['predicted_points'].sum().reset_index()
    projected_points.columns = ['driverId', 'projected_points']
    
    # Merge with current standings
    final_projection = current_standings.merge(projected_points, on='driverId', how='outer')
    final_projection['projected_points'] = final_projection['projected_points'].fillna(0)
    final_projection['total_points'] = final_projection['total_points'].fillna(0)
    final_projection['final_projected_points'] = final_projection['total_points'] + final_projection['projected_points']
    
    # Sort by projected final points
    final_projection = final_projection.sort_values('final_projected_points', ascending=False).reset_index(drop=True)
    final_projection['projected_position'] = range(1, len(final_projection) + 1)
    
    # Add driver names if not already present
    if 'driver_name' not in final_projection.columns:
        driver_names = drivers[['driverId', 'forename', 'surname']].copy()
        driver_names['driver_name'] = driver_names['forename'] + ' ' + driver_names['surname']
        
        final_projection = final_projection.merge(
            driver_names[['driverId', 'driver_name']], 
            on='driverId', 
            how='left'
        )
        final_projection['driver_name'] = final_projection['driver_name'].fillna('Unknown Driver')
    
    print("\n2025 Championship Projection (Top 20):")
    print(final_projection[['projected_position', 'driver_name', 'total_points', 'projected_points', 'final_projected_points']].head(20).to_string(index=False))
    
    # Save projection
    final_projection.to_csv('2025_championship_projection.csv', index=False)
    print("\nSaved championship projection to: 2025_championship_projection.csv")
else:
    print("\nNo remaining 2025 races to project.")

# ========================
# Step 10: Save Artifacts
# ========================

print("\nStep 10: Saving artifacts...")

# Save training data
train_df.to_parquet('driver_race_prerace_train.parquet', index=False)
print("Saved training data to: driver_race_prerace_train.parquet")

# Save CV results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.to_csv('cv_results.csv', index=False)
print("Saved CV results to: cv_results.csv")

# Create and save metrics summary
metrics_summary = {
    'best_params': grid_search.best_params_,
    'best_cv_logloss': -grid_search.best_score_,
    'train_logloss': train_logloss,
    'train_macro_f1': train_f1,
    'loocv_logloss': loocv_logloss if loocv_predictions else None,
    'training_samples': len(train_df),
    'holdout_samples': len(holdout_df)
}

import json
with open('metrics_summary.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2)
print("Saved metrics summary to: metrics_summary.json")

# Create visualization of CV results
plt.figure(figsize=(12, 6))

# Plot 1: Parameter comparison
plt.subplot(1, 2, 1)
cv_results_sorted = cv_results.sort_values('mean_test_score', ascending=False).head(10)
plt.barh(range(len(cv_results_sorted)), -cv_results_sorted['mean_test_score'])
plt.yticks(range(len(cv_results_sorted)), 
           [f"n={row['param_classifier__n_neighbors']}, w={row['param_classifier__weights'][:3]}, m={row['param_classifier__metric'][:3]}" 
            for _, row in cv_results_sorted.iterrows()])
plt.xlabel('Negative Log Loss (lower is better)')
plt.title('Top 10 Hyperparameter Combinations')
plt.tight_layout()

# Plot 2: Predicted vs Actual Championship Standings (if available)
if len(holdout_df) > 0:
    plt.subplot(1, 2, 2)
    top_drivers = final_projection.head(10)
    plt.barh(range(len(top_drivers)), top_drivers['final_projected_points'])
    plt.yticks(range(len(top_drivers)), top_drivers['driver_name'])
    plt.xlabel('Projected Final Points')
    plt.title('Top 10 Projected Championship Standings')
    plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('model_results.png', dpi=150, bbox_inches='tight')
print("Saved visualization to: model_results.png")

print("\n" + "="*50)
print("Pipeline complete!")
print("="*50)
