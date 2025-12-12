"""
F1 Championship Prediction Pipeline - XGBoost POSITION-BASED
Predicts finishing position (1st, 2nd, 3rd, etc.) using XGBoost
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
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
# Step 8: Build XGBoost REGRESSION Pipeline
# ========================

print("\nStep 8: Building XGBoost regression pipeline...")

# Define feature columns with importance groupings
categorical_features = ['circuitId']
passthrough_features = ['is_sprint_weekend']

# Feature groupings based on importance
# Track history: 2.5x weight
track_features = [
    'drv_track_prev_avg_points',
    'drv_track_prev_starts',
]

# Recent form: 2.0x weight
recent_form_features = [
    'drv_last3_avg_points',
    'drv_last3_avg_grid',
    'drv_last3_top10_rate',
]

# Grid position: 1.5x weight
grid_features = [
    'grid',
]

# Season standing: 1.5x weight
season_standing_features = [
    'cum_points_before',
    'position_before',
    'wins_before',
]

# Other features: 1.0x weight
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
    track_weight=2.5,
    recent_weight=2.0,
    grid_weight=1.5,
    season_weight=1.5,
    other_weight=1.0
)

# Create pipeline with XGBoost REGRESSOR
pipeline = Pipeline([
    ('preprocessor', weighted_preprocessor),
    ('regressor', XGBRegressor(random_state=42, enable_categorical=False))
])

# Define parameter grid for XGBoost
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [3, 5, 7],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__subsample': [0.8, 1.0],
    'regressor__colsample_bytree': [0.8, 1.0],
    'regressor__min_child_weight': [1, 3, 5]
}

# Prepare features and target
X_train = train_df[categorical_features + passthrough_features + numeric_features]
y_train = train_df['target_position']

print(f"Feature matrix shape: {X_train.shape}")
print(f"Target distribution (top 10 positions):")
print(y_train.value_counts().sort_index().head(10))

# Grid search with MAE scoring
print("\nPerforming hyperparameter tuning with 5-fold CV...")
print("(This may take several minutes due to XGBoost's extensive parameter space)")
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
joblib.dump(best_model, 'trained_xgboost_position_model.joblib')
print("Saved trained model to: trained_xgboost_position_model.joblib")

# Save CV results for comparison
cv_results_df = pd.DataFrame(grid_search.cv_results_)
cv_results_df.to_csv('xgboost_cv_results.csv', index=False)
print("Saved CV results to: xgboost_cv_results.csv")

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
    
    # Combine all predictions
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    # Merge with driver names
    predictions_df = predictions_df.merge(
        drivers[['driverId', 'forename', 'surname']], 
        on='driverId', 
        how='left'
    )
    predictions_df['driver_name'] = predictions_df['forename'] + ' ' + predictions_df['surname']
    
    # Save predictions
    output_cols = ['driver_name', 'round', 'predicted_position', 'predicted_points', 'is_sprint_weekend']
    predictions_df[output_cols].to_csv('xgboost_remaining_races_predictions.csv', index=False)
    print(f"\nSaved predictions to: xgboost_remaining_races_predictions.csv")
    
    # Calculate final championship standings
    predicted_total_points = predictions_df.groupby('driverId')['predicted_points'].sum().reset_index()
    predicted_total_points.columns = ['driverId', 'predicted_points']
    
    final_standings = current_standings.merge(predicted_total_points, on='driverId', how='left')
    final_standings['predicted_points'] = final_standings['predicted_points'].fillna(0)
    final_standings['projected_total'] = final_standings['total_points'] + final_standings['predicted_points']
    final_standings = final_standings.sort_values('projected_total', ascending=False).reset_index(drop=True)
    final_standings['projected_position'] = range(1, len(final_standings) + 1)
    
    # Save championship projection
    final_standings[['driver_name', 'total_points', 'predicted_points', 'projected_total', 'projected_position']].to_csv(
        'xgboost_2025_championship_projection.csv', index=False
    )
    print(f"\nSaved championship projection to: xgboost_2025_championship_projection.csv")
    
    print("\n2025 Championship Projection (Top 10):")
    print(final_standings[['driver_name', 'total_points', 'predicted_points', 'projected_total', 'projected_position']].head(10).to_string(index=False))

else:
    print("\nNo remaining races to predict.")

print("\n" + "="*50)
print("XGBoost Pipeline Complete!")
print("="*50)
