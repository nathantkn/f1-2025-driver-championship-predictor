# F1 Championship Prediction - Pipeline Results Summary

## Overview
This pipeline implements a K-Nearest Neighbors (K-NN) classification model to predict F1 race finishing positions and project the 2025 championship.

## Model Configuration

### Best Hyperparameters (from Grid Search)
- **n_neighbors**: 31
- **weights**: uniform
- **metric**: euclidean

### Features Used
1. **Categorical Features**:
   - circuitId (one-hot encoded)

2. **Passthrough Features**:
   - is_sprint_weekend

3. **Numeric Features** (standardized):
   - grid (starting position)
   - cum_points_before (cumulative points before this race)
   - position_before (championship position before this race)
   - wins_before (wins before this race)
   - drv_last3_avg_points (driver's avg points in last 3 races)
   - drv_last3_top10_rate (driver's top 10 finish rate in last 3 races)
   - drv_last3_dnf_rate (driver's DNF rate in last 3 races)
   - drv_last3_avg_grid (driver's avg grid position in last 3 races)
   - tm_last3_avg_points (teammate's avg points in last 3 races)
   - tm_last3_top10_rate (teammate's top 10 rate in last 3 races)
   - drv_track_prev_starts (driver's previous starts at this circuit)
   - drv_track_prev_avg_points (driver's avg points at this circuit)

## Performance Metrics

### Cross-Validation (5-fold Stratified)
- **Log-loss**: 1.6935

### Training Set Performance
- **Log-loss**: 0.9945
- **Macro F1-score**: 0.3483

### Walk-Forward LOOCV (by season/round)
- **Log-loss**: 1.6607

## Training Data
- **Total samples**: 6,850 race entries
- **Time period**: 2010-2025 (up to round 22)
- **Target classes**: 6 finish position buckets
  - 0 points (no finish/low position)
  - 1–3 points
  - 4–6 points
  - 7–10 points
  - 12–15 points
  - 18–26 points (podium/win)

## Target Distribution
- 0 points: 3,605 samples (52.6%)
- 1–3 points: 654 samples (9.5%)
- 7–10 points: 653 samples (9.5%)
- 18–26 points: 652 samples (9.5%)
- 4–6 points: 646 samples (9.4%)
- 12–15 points: 640 samples (9.3%)

## 2025 Championship Status
All 22 rounds of the 2025 season have been completed. The current championship standings are:

| Position | Driver ID | Total Points |
|----------|-----------|--------------|
| 1 | 846 | 367.0 |
| 2 | 857 | 345.0 |
| 3 | 830 | 339.0 |
| 4 | 847 | 271.0 |
| 5 | 844 | 209.0 |
| 6 | 1 | 131.0 |
| 7 | 863 | 125.0 |
| 8 | 848 | 70.0 |
| 9 | 864 | 50.0 |
| 10 | 807 | 49.0 |

## Leakage Prevention Measures
✅ All rolling features computed with `.shift(1)` before rolling
✅ Standings joined from `prev_raceId` (never current race)
✅ Only pre-race features used (grid position allowed)
✅ No current race results used in feature engineering

## Saved Artifacts
- `trained_knn_pipeline.joblib` - Trained scikit-learn pipeline
- `driver_race_prerace_train.parquet` - Training dataset
- `cv_results.csv` - Complete cross-validation results
- `metrics_summary.json` - Model performance metrics
- `model_results.png` - Visualization of results

## Model Interpretation
The K-NN model with 31 neighbors provides a balanced approach to classification:
- Uses uniform weighting (all 31 nearest neighbors vote equally)
- Euclidean distance metric for similarity measurement
- Walk-forward validation log-loss of 1.66 indicates reasonable out-of-sample performance
- The gap between training (0.99) and validation (1.69) log-loss suggests some overfitting, but the walk-forward validation confirms generalization capability

## Next Steps (If Future Races Exist)
To project future races:
1. Update `base.csv` with pre-race features for upcoming races
2. Run the pipeline with the saved model
3. Generate expected points from probability predictions
4. Project final championship standings

## Notes
- The model was trained on data from 2010-2025 (22 rounds completed)
- No holdout races remain for 2025 projection
- The pipeline is ready to be used for 2026 season predictions when data becomes available
