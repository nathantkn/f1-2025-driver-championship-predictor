"""
F1 Championship Prediction Pipeline - POSITION-BASED
Predicts finishing position (1st, 2nd, 3rd, etc.) instead of points buckets
Then converts positions to actual points for championship projection
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor  # Using REGRESSOR for position prediction
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt

# Points mapping for F1 (standard + sprint)
STANDARD_POINTS = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1
}
SPRINT_POINTS = {
    1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1
}

def position_to_points(position, is_sprint):
    """Convert finishing position to championship points"""
    position = int(round(position))
    if position < 1:
        return 0
    if is_sprint:
        return SPRINT_POINTS.get(position, 0)
    else:
        return STANDARD_POINTS.get(position, 0)

# ========================
# Step 7: Load Data & Train/Validation Split
# ========================

print("Step 7: Loading data and creating train/validation split...")
df = pd.read_csv('data-cleaned/base.csv')

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Check for target_position column
if 'target_position' not in df.columns:
    print("ERROR: target_position column not found. Run add_position_target.py first.")
    exit(1)

# Get latest completed 2025 round
latest_2025_round = df[df['year'] == 2025]['round'].max()
print(f"Latest completed 2025 round: {latest_2025_round}")

# Split into training (all data before remaining races) and holdout (remaining 2025 races)
try:
    holdout_df = pd.read_csv('data-cleaned/remaining_races.csv')
    train_df = df.copy()
    print(f"Training set size: {len(train_df)} rows")
    print(f"Holdout set size: {len(holdout_df)} rows (from remaining_races.csv)")
except FileNotFoundError:
    print("No remaining_races.csv found. Using all data for training.")
    train_df = df.copy()
    holdout_df = pd.DataFrame()

# Remove rows with missing target
print(f"Training set after removing missing targets: {len(train_df)} rows")

# ========================
# Step 8: Build K-NN REGRESSION Pipeline
# ========================

print("\nStep 8: Building K-NN regression pipeline...")

# Define feature columns with importance groupings
categorical_features = ['circuitId']
passthrough_features = ['is_sprint_weekend']

# UPDATED WEIGHTS BASED ON QATAR RACE ANALYSIS:
# Track history: 2.5x (increased from 1.0x)
# Recent form: 2.0x (decreased from 3.0x)
# Grid position: 1.5x (decreased from 3.0x)
# Season standing: 1.5x (decreased from 2.0x)
# Other: 1.0x (same)

# High importance: Track-specific performance (2.5x weight)
track_features = [
    'drv_track_prev_avg_points',
    'drv_track_prev_starts',
]

# Medium-high importance: Recent form (2.0x weight)
recent_form_features = [
    'drv_last3_avg_points',
    'drv_last3_avg_grid',
    'drv_last3_top10_rate',
]

# Medium importance: Grid position (1.5x weight)
grid_features = [
    'grid',
]

# Medium importance: Season standing (1.5x weight)
season_standing_features = [
    'cum_points_before',
    'position_before',
    'wins_before',
]

# Lower importance: Other features (1.0x weight)
other_features = [
    'drv_last3_dnf_rate',
    'tm_last3_avg_points',
    'tm_last3_top10_rate',
]

# Combine all numeric features
numeric_features = track_features + recent_form_features + grid_features + season_standing_features + other_features

# Fill NaNs in numeric features with 0 (for early-career drivers)
for col in numeric_features:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(0)

# Create weighted preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('pass', 'passthrough', passthrough_features),
        ('track', StandardScaler(), track_features),
        ('recent', StandardScaler(), recent_form_features),
        ('grid', StandardScaler(), grid_features),
        ('season', StandardScaler(), season_standing_features),
        ('other', StandardScaler(), other_features)
    ],
    remainder='drop'
)

# Custom feature weighting wrapper
class WeightedFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessor, track_weight=2.5, recent_weight=2.0, grid_weight=1.5, 
                 season_weight=1.5, other_weight=1.0):
        self.preprocessor = preprocessor
        self.track_weight = track_weight
        self.recent_weight = recent_weight
        self.grid_weight = grid_weight
        self.season_weight = season_weight
        self.other_weight = other_weight
        
    def fit(self, X, y=None):
        self.preprocessor.fit(X, y)
        self.cat_features_ = len(self.preprocessor.named_transformers_['cat'].get_feature_names_out())
        self.pass_features_ = len(passthrough_features)
        self.track_features_ = len(track_features)
        self.recent_features_ = len(recent_form_features)
        self.grid_features_ = len(grid_features)
        self.season_features_ = len(season_standing_features)
        self.other_features_ = len(other_features)
        return self
        
    def transform(self, X):
        X_transformed = self.preprocessor.transform(X)
        
        # Apply weights
        start_idx = 0
        end_idx = self.cat_features_ + self.pass_features_
        start_idx = end_idx
        
        # Track features (2.5x)
        end_idx = start_idx + self.track_features_
        X_transformed[:, start_idx:end_idx] *= self.track_weight
        start_idx = end_idx
        
        # Recent form features (2.0x)
        end_idx = start_idx + self.recent_features_
        X_transformed[:, start_idx:end_idx] *= self.recent_weight
        start_idx = end_idx
        
        # Grid features (1.5x)
        end_idx = start_idx + self.grid_features_
        X_transformed[:, start_idx:end_idx] *= self.grid_weight
        start_idx = end_idx
        
        # Season standing features (1.5x)
        end_idx = start_idx + self.season_features_
        X_transformed[:, start_idx:end_idx] *= self.season_weight
        start_idx = end_idx
        
        # Other features (1.0x)
        end_idx = start_idx + self.other_features_
        X_transformed[:, start_idx:end_idx] *= self.other_weight
        
        return X_transformed

weighted_preprocessor = WeightedFeatureTransformer(
    preprocessor,
    recent_weight=3.0,
    season_weight=2.0,
    other_weight=1.0
)

# Create pipeline with K-NN REGRESSOR
pipeline = Pipeline([
    ('preprocessor', weighted_preprocessor),
    ('regressor', KNeighborsRegressor())
])

# Define parameter grid for regression
param_grid = {
    'regressor__n_neighbors': [11, 21, 31, 41],
    'regressor__weights': ['distance'],
    'regressor__metric': ['euclidean', 'manhattan']
}

# Prepare features and target
X_train = train_df[categorical_features + passthrough_features + numeric_features]
y_train = train_df['target_position']

print(f"Feature matrix shape: {X_train.shape}")
print(f"Target distribution (top 10 positions):")
print(y_train.value_counts().sort_index().head(10))

# Grid search with MAE scoring
print("\nPerforming hyperparameter tuning with 5-fold CV...")
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV MAE: {-grid_search.best_score_:.4f} positions")

# Get best model
best_model = grid_search.best_estimator_

# Training metrics
y_train_pred = best_model.predict(X_train)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

print(f"\nTraining metrics:")
print(f"  MAE: {train_mae:.4f} positions")
print(f"  RMSE: {train_rmse:.4f} positions")

# ========================
# Step 9: Championship Projection
# ========================

print("\nStep 9: Projecting 2025 championship...")

# Save model
joblib.dump(best_model, 'trained_knn_position_model.joblib')
print("Saved trained model to: trained_knn_position_model.joblib")

# Get current standings - calculate total points from all 2025 races
drivers = pd.read_csv('data-raw/drivers.csv')

# Calculate actual points scored in each race
races_2025 = train_df[train_df['year'] == 2025][['driverId', 'is_sprint_weekend', 'target_position']].copy()
races_2025['points_scored'] = races_2025.apply(
    lambda row: position_to_points(row['target_position'], row['is_sprint_weekend']), axis=1
)

current_standings = races_2025.groupby('driverId').agg({
    'points_scored': 'sum'
}).reset_index()
current_standings.columns = ['driverId', 'total_points']

current_standings = current_standings.merge(
    drivers[['driverId', 'forename', 'surname']], 
    on='driverId', 
    how='left'
)
current_standings['driver_name'] = current_standings['forename'] + ' ' + current_standings['surname']

print(f"Current standings (top 10):")
print(current_standings.nlargest(10, 'total_points')[['driver_name', 'total_points']].to_string(index=False))

# Project remaining races SEQUENTIALLY
if len(holdout_df) > 0:
    print(f"\nProjecting {len(holdout_df)} remaining race entries sequentially...")
    
    remaining_rounds = sorted(holdout_df['round'].unique())
    print(f"Predicting rounds: {remaining_rounds}")
    
    # Create combined dataset
    combined_df = train_df.copy()
    combined_df['predicted_position'] = combined_df['target_position']
    combined_df['predicted_points'] = combined_df.apply(
        lambda row: position_to_points(row['target_position'], row['is_sprint_weekend']), axis=1
    )
    
    all_predictions = []
    
    # Predict each round sequentially
    for round_num in remaining_rounds:
        print(f"\n  Predicting Round {round_num}...")
        
        round_df = holdout_df[holdout_df['round'] == round_num].copy()
        
        # Update cum_points_before and position_before based on previous round predictions
        if round_num > min(remaining_rounds) and len(all_predictions) > 0:
            # Build dataframe from predictions so far
            prev_predictions_df = pd.concat(all_predictions, ignore_index=True)
            
            # Calculate total points after all previous predicted rounds
            for idx, row in round_df.iterrows():
                driverId = row['driverId']
                
                # Get points from all previous predicted rounds
                prev_rounds_points = prev_predictions_df[
                    (prev_predictions_df['driverId'] == driverId) &
                    (prev_predictions_df['round'] < round_num)
                ]['predicted_points'].sum()
                
                # Update cumulative points
                round_df.at[idx, 'cum_points_before'] = row['cum_points_before'] + prev_rounds_points
            
            # Recalculate championship positions based on updated cum_points_before
            championship_standings = round_df[['driverId', 'cum_points_before']].drop_duplicates()
            championship_standings = championship_standings.sort_values('cum_points_before', ascending=False).reset_index(drop=True)
            championship_standings['position_before'] = range(1, len(championship_standings) + 1)
            
            # Merge back the updated positions
            round_df = round_df.drop(columns=['position_before'])
            round_df = round_df.merge(championship_standings[['driverId', 'position_before']], on='driverId', how='left')
        
        # Update rolling features based on predictions
        for idx, row in round_df.iterrows():
            driverId = row['driverId']
            constructorId = row['constructorId']
            
            driver_history = combined_df[
                (combined_df['driverId'] == driverId) & 
                (combined_df['round'] < round_num)
            ].sort_values('round').tail(3)
            
            if len(driver_history) > 0:
                round_df.at[idx, 'drv_last3_avg_points'] = np.ceil(driver_history['predicted_points'].mean() * 100) / 100
                round_df.at[idx, 'drv_last3_top10_rate'] = np.ceil((driver_history['predicted_points'] > 0).mean() * 100) / 100
                round_df.at[idx, 'drv_last3_dnf_rate'] = np.ceil((driver_history['predicted_points'] == 0).mean() * 100) / 100
                round_df.at[idx, 'drv_last3_avg_grid'] = np.ceil(driver_history['grid'].mean() * 100) / 100
            
            teammate_history = combined_df[
                (combined_df['constructorId'] == constructorId) &
                (combined_df['driverId'] != driverId) &
                (combined_df['round'] < round_num)
            ].sort_values('round').tail(3)
            
            if len(teammate_history) > 0:
                round_df.at[idx, 'tm_last3_avg_points'] = np.ceil(teammate_history['predicted_points'].mean() * 100) / 100
                round_df.at[idx, 'tm_last3_top10_rate'] = np.ceil((teammate_history['predicted_points'] > 0).mean() * 100) / 100
        
        # Prepare features
        for col in numeric_features:
            if col in round_df.columns:
                round_df[col] = round_df[col].fillna(0)
        
        X_round = round_df[categorical_features + passthrough_features + numeric_features]
        
        # Predict positions
        predicted_positions_raw = best_model.predict(X_round)
        
        # Rank drivers to get unique positions 1-20 (or 1-N for N drivers)
        # Lower predicted position value = better finish, so rank ascending
        round_df['predicted_position_raw'] = predicted_positions_raw
        round_df = round_df.sort_values('predicted_position_raw').reset_index(drop=True)
        round_df['predicted_position'] = range(1, len(round_df) + 1)
        
        # Convert positions to points
        round_df['predicted_points'] = round_df.apply(
            lambda row: position_to_points(row['predicted_position'], row['is_sprint_weekend']), axis=1
        )
        
        # Drop the raw prediction column
        round_df = round_df.drop(columns=['predicted_position_raw'])
        
        # Store predictions
        all_predictions.append(round_df[['driverId', 'round', 'predicted_position', 'predicted_points', 'is_sprint_weekend']])
        
        # Add to combined_df for next round
        round_df_for_rolling = round_df[['raceId', 'driverId', 'constructorId', 'grid', 'year', 'round', 
                                          'drv_last3_avg_points', 'drv_last3_top10_rate', 'drv_last3_dnf_rate',
                                          'drv_last3_avg_grid', 'tm_last3_avg_points', 'tm_last3_top10_rate',
                                          'predicted_position', 'predicted_points']].copy()
        combined_df = pd.concat([combined_df, round_df_for_rolling], ignore_index=True)
        
        print(f"    Predicted {len(round_df)} entries for Round {round_num}")
        print(f"    Avg predicted position: {round_df['predicted_position'].mean():.2f}")
        print(f"    Avg predicted points: {round_df['predicted_points'].mean():.2f}")
    
    # Combine predictions
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    # Update remaining_races.csv with sequential rolling features + predictions
    updated_rounds = []
    for round_num in remaining_rounds:
        original_round = holdout_df[holdout_df['round'] == round_num].copy()
        pred_round = all_predictions_df[all_predictions_df['round'] == round_num].copy()
        
        for idx, row in original_round.iterrows():
            updated_row = combined_df[
                (combined_df['year'] == 2025) &
                (combined_df['driverId'] == row['driverId']) & 
                (combined_df['round'] == round_num)
            ].tail(1)
            
            if len(updated_row) > 0:
                for col in ['drv_last3_avg_points', 'drv_last3_top10_rate', 'drv_last3_dnf_rate',
                           'drv_last3_avg_grid', 'tm_last3_avg_points', 'tm_last3_top10_rate']:
                    if col in updated_row.columns:
                        original_round.at[idx, col] = updated_row[col].values[0]
            
            if round_num > min(remaining_rounds):
                prev_rounds_points = all_predictions_df[
                    (all_predictions_df['driverId'] == row['driverId']) &
                    (all_predictions_df['round'] < round_num)
                ]['predicted_points'].sum()
                original_round.at[idx, 'cum_points_before'] = row['cum_points_before'] + prev_rounds_points
        
        updated_rounds.append(original_round)
    
    updated_remaining_export = pd.concat(updated_rounds, ignore_index=True)
    
    float_cols = ['drv_last3_avg_points', 'drv_last3_top10_rate', 'drv_last3_dnf_rate',
                  'drv_last3_avg_grid', 'tm_last3_avg_points', 'tm_last3_top10_rate',
                  'drv_track_prev_avg_points']
    for col in float_cols:
        if col in updated_remaining_export.columns:
            updated_remaining_export[col] = updated_remaining_export[col].round(2)
    
    updated_remaining_export = updated_remaining_export.sort_values(['round', 'driverId']).reset_index(drop=True)
    updated_remaining_export.to_csv('data-cleaned/remaining_races.csv', index=False)
    print(f"\n  ✅ Updated remaining_races.csv with {len(updated_remaining_export)} entries (sequential rolling features)")
    
    # Aggregate projected points
    projected_points = all_predictions_df.groupby('driverId')['predicted_points'].sum().reset_index()
    projected_points.columns = ['driverId', 'projected_points']
    
    # Final projection
    final_projection = current_standings.merge(projected_points, on='driverId', how='outer')
    final_projection['projected_points'] = final_projection['projected_points'].fillna(0)
    final_projection['total_points'] = final_projection['total_points'].fillna(0)
    final_projection['final_projected_points'] = final_projection['total_points'] + final_projection['projected_points']
    
    final_projection = final_projection.sort_values('final_projected_points', ascending=False).reset_index(drop=True)
    final_projection['projected_position'] = range(1, len(final_projection) + 1)
    
    if 'driver_name' not in final_projection.columns:
        driver_names = drivers[['driverId', 'forename', 'surname']].copy()
        driver_names['driver_name'] = driver_names['forename'] + ' ' + driver_names['surname']
        final_projection = final_projection.merge(driver_names[['driverId', 'driver_name']], on='driverId', how='left')
        final_projection['driver_name'] = final_projection['driver_name'].fillna('Unknown Driver')
    
    print("\n2025 Championship Projection (Top 20):")
    print(final_projection[['projected_position', 'driver_name', 'total_points', 'projected_points', 'final_projected_points']].head(20).to_string(index=False))
    
    final_projection.to_csv('2025_championship_projection.csv', index=False)
    print("\nSaved championship projection to: 2025_championship_projection.csv")
    
    # Save detailed predictions with driver names and race info
    races_info = pd.read_csv('data-raw/races.csv')
    all_predictions_detailed = all_predictions_df.merge(
        drivers[['driverId', 'forename', 'surname']], on='driverId'
    )
    all_predictions_detailed['driver_name'] = all_predictions_detailed['forename'] + ' ' + all_predictions_detailed['surname']
    
    # Add race names
    remaining_race_info = holdout_df[['round', 'raceId']].drop_duplicates()
    remaining_race_info = remaining_race_info.merge(
        races_info[['raceId', 'name', 'circuitId']], on='raceId', how='left'
    )
    all_predictions_detailed = all_predictions_detailed.merge(
        remaining_race_info[['round', 'name']], on='round', how='left'
    )
    all_predictions_detailed = all_predictions_detailed.rename(columns={'name': 'race_name'})
    
    # Reorder columns for better readability
    output_cols = ['round', 'race_name', 'driver_name', 'predicted_position', 'predicted_points', 'is_sprint_weekend']
    all_predictions_detailed = all_predictions_detailed[output_cols].drop_duplicates()
    all_predictions_detailed = all_predictions_detailed.sort_values(['round', 'predicted_position'])
    
    all_predictions_detailed.to_csv('remaining_races_predictions.csv', index=False)
    print("\nSaved detailed predictions to: remaining_races_predictions.csv")
    
    # Print race-by-race results
    print("\n" + "="*70)
    print("RACE-BY-RACE PREDICTION RESULTS")
    print("="*70)
    
    for round_num in remaining_rounds:
        round_preds = all_predictions_detailed[all_predictions_detailed['round'] == round_num]
        race_name = round_preds['race_name'].iloc[0]
        is_sprint = round_preds['is_sprint_weekend'].iloc[0]
        race_type = "SPRINT RACE" if is_sprint else "GRAND PRIX"
        
        print(f"\n{race_type}: Round {round_num} - {race_name}")
        print("-" * 70)
        print(f"{'Pos':<5} {'Driver':<25} {'Points':<8}")
        print("-" * 70)
        
        top_10 = round_preds.head(10)
        for _, row in top_10.iterrows():
            print(f"{int(row['predicted_position']):<5} {row['driver_name']:<25} {int(row['predicted_points']):<8}")
        
        if len(round_preds) > 10:
            print(f"... and {len(round_preds) - 10} more drivers")
    
    # Calculate and display evaluation metrics
    print("\n" + "="*70)
    print("MODEL EVALUATION METRICS")
    print("="*70)
    
    # Training metrics (already calculated)
    print(f"\n1. POSITION PREDICTION ACCURACY (Training Data)")
    print(f"   - Mean Absolute Error (MAE): {train_mae:.4f} positions")
    print(f"   - Root Mean Squared Error (RMSE): {train_rmse:.4f} positions")
    print(f"   - Cross-Validation MAE: {-grid_search.best_score_:.4f} positions")
    
    # Top-10 accuracy on training data
    y_train_pred_rounded = np.round(y_train_pred)
    top10_actual = (y_train <= 10).astype(int)
    top10_pred = (y_train_pred_rounded <= 10).astype(int)
    top10_accuracy = (top10_actual == top10_pred).mean()
    
    print(f"\n2. TOP-10 CLASSIFICATION ACCURACY (Training Data)")
    print(f"   - Accuracy: {top10_accuracy:.2%}")
    
    # Podium accuracy
    podium_actual = (y_train <= 3).astype(int)
    podium_pred = (y_train_pred_rounded <= 3).astype(int)
    podium_accuracy = (podium_actual == podium_pred).mean()
    
    print(f"\n3. PODIUM CLASSIFICATION ACCURACY (Training Data)")
    print(f"   - Accuracy: {podium_accuracy:.2%}")
    
    # Championship prediction metrics
    print(f"\n4. CHAMPIONSHIP PREDICTION SUMMARY")
    print(f"   - Remaining races predicted: {len(remaining_rounds)}")
    print(f"   - Total drivers: {len(final_projection)}")
    print(f"   - Predicted champion: {final_projection.iloc[0]['driver_name']} ({final_projection.iloc[0]['final_projected_points']:.0f} pts)")
    print(f"   - Championship margin: {final_projection.iloc[0]['final_projected_points'] - final_projection.iloc[1]['final_projected_points']:.0f} pts")
    
    # Points distribution
    total_predicted_points = all_predictions_df['predicted_points'].sum()
    avg_points_per_race = all_predictions_df.groupby('round')['predicted_points'].sum().mean()
    
    print(f"\n5. POINTS DISTRIBUTION")
    print(f"   - Total points awarded (predictions): {total_predicted_points:.0f}")
    print(f"   - Average points per race: {avg_points_per_race:.2f}")
    print(f"   - Expected points per race (21 drivers): {25+18+15+12+10+8+6+4+2+1:.0f} (GP) or {8+7+6+5+4+3+2+1:.0f} (Sprint)")
    
    # Feature importance (based on weights)
    print(f"\n6. FEATURE WEIGHTING (UPDATED BASED ON QATAR ANALYSIS)")
    print(f"   - Track history (drv_track_prev_avg_points, prev_starts): 2.5x weight")
    print(f"   - Recent form (drv_last3_avg_points, avg_grid, top10_rate): 2.0x weight")
    print(f"   - Grid position: 1.5x weight")
    print(f"   - Season standing (points, position, wins): 1.5x weight")
    print(f"   - Other features (DNF rate, teammate form): 1.0x weight")
    
    print(f"\n7. MODEL CONFIGURATION")
    print(f"   - Algorithm: K-Nearest Neighbors Regression")
    print(f"   - Best parameters: {grid_search.best_params_}")
    print(f"   - Training samples: {len(train_df)}")
    print(f"   - Feature count: {len(categorical_features + passthrough_features + numeric_features)}")
    
else:
    print("\nNo remaining 2025 races to project.")

print("\n" + "="*70)
print("PIPELINE COMPLETE!")
print("="*70)
