"""
Model Comparison Script - KNN vs XGBoost
Compares metrics between KNN and XGBoost models for F1 position prediction
"""

import pandas as pd
import json
import os

# Change to project root if running from scripts directory
if os.path.basename(os.getcwd()) == "scripts":
    os.chdir("..")

print("="*70)
print("F1 POSITION PREDICTION MODEL COMPARISON")
print("KNN vs XGBoost")
print("="*70)

# Load metrics from both models
try:
    with open("knn_metrics_summary.json", "r") as f:
        knn_metrics = json.load(f)
    print("\n✓ Loaded KNN metrics")
except FileNotFoundError:
    print("\n✗ KNN metrics not found. Please run evaluation_model.py first.")
    knn_metrics = None

try:
    with open("xgboost_metrics_summary.json", "r") as f:
        xgb_metrics = json.load(f)
    print("✓ Loaded XGBoost metrics")
except FileNotFoundError:
    print("✗ XGBoost metrics not found. Please run evaluation_xgboost_model.py first.")
    xgb_metrics = None

if knn_metrics is None or xgb_metrics is None:
    print("\nCannot proceed without both metrics files.")
    exit(1)

print("\n" + "-"*70)
print("SPRINT RACE (Round 23) COMPARISON")
print("-"*70)

sprint_comparison = pd.DataFrame({
    'Metric': ['MAE (points)', 'RMSE (points)', 'MSE', 'Top-8 Accuracy (%)'],
    'KNN': [
        knn_metrics['sprint']['MAE'],
        knn_metrics['sprint']['RMSE'],
        knn_metrics['sprint']['MSE'],
        knn_metrics['sprint']['Top8_Accuracy']
    ],
    'XGBoost': [
        xgb_metrics['sprint']['MAE'],
        xgb_metrics['sprint']['RMSE'],
        xgb_metrics['sprint']['MSE'],
        xgb_metrics['sprint']['Top8_Accuracy']
    ]
})

# Calculate improvement
sprint_comparison['Difference'] = sprint_comparison['XGBoost'] - sprint_comparison['KNN']
sprint_comparison['% Change'] = ((sprint_comparison['XGBoost'] - sprint_comparison['KNN']) / 
                                  sprint_comparison['KNN'] * 100).round(1)

# For error metrics, negative change is better; for accuracy, positive is better
sprint_comparison['Winner'] = sprint_comparison.apply(
    lambda row: '🏆 XGBoost' if (row['Metric'] == 'Top-8 Accuracy (%)' and row['Difference'] > 0) or 
                                  (row['Metric'] != 'Top-8 Accuracy (%)' and row['Difference'] < 0)
                else '🏆 KNN' if row['Difference'] != 0 else '🤝 Tie',
    axis=1
)

print("\n" + sprint_comparison.to_string(index=False))

print("\n" + "-"*70)
print("GRAND PRIX (Round 24) COMPARISON")
print("-"*70)

gp_comparison = pd.DataFrame({
    'Metric': ['MAE (points)', 'RMSE (points)', 'MSE', 'Top-10 Accuracy (%)'],
    'KNN': [
        knn_metrics['grand_prix']['MAE'],
        knn_metrics['grand_prix']['RMSE'],
        knn_metrics['grand_prix']['MSE'],
        knn_metrics['grand_prix']['Top10_Accuracy']
    ],
    'XGBoost': [
        xgb_metrics['grand_prix']['MAE'],
        xgb_metrics['grand_prix']['RMSE'],
        xgb_metrics['grand_prix']['MSE'],
        xgb_metrics['grand_prix']['Top10_Accuracy']
    ]
})

# Calculate improvement
gp_comparison['Difference'] = gp_comparison['XGBoost'] - gp_comparison['KNN']
gp_comparison['% Change'] = ((gp_comparison['XGBoost'] - gp_comparison['KNN']) / 
                              gp_comparison['KNN'] * 100).round(1)

# For error metrics, negative change is better; for accuracy, positive is better
gp_comparison['Winner'] = gp_comparison.apply(
    lambda row: '🏆 XGBoost' if (row['Metric'] == 'Top-10 Accuracy (%)' and row['Difference'] > 0) or 
                                  (row['Metric'] != 'Top-10 Accuracy (%)' and row['Difference'] < 0)
                else '🏆 KNN' if row['Difference'] != 0 else '🤝 Tie',
    axis=1
)

print("\n" + gp_comparison.to_string(index=False))

print("\n" + "="*70)
print("OVERALL SUMMARY")
print("="*70)

# Count wins
knn_sprint_wins = (sprint_comparison['Winner'] == '🏆 KNN').sum()
xgb_sprint_wins = (sprint_comparison['Winner'] == '🏆 XGBoost').sum()
knn_gp_wins = (gp_comparison['Winner'] == '🏆 KNN').sum()
xgb_gp_wins = (gp_comparison['Winner'] == '🏆 XGBoost').sum()

print(f"\nSprint Race Metrics:")
print(f"  KNN wins: {knn_sprint_wins}/4 metrics")
print(f"  XGBoost wins: {xgb_sprint_wins}/4 metrics")

print(f"\nGrand Prix Metrics:")
print(f"  KNN wins: {knn_gp_wins}/4 metrics")
print(f"  XGBoost wins: {xgb_gp_wins}/4 metrics")

print(f"\nTotal:")
print(f"  KNN wins: {knn_sprint_wins + knn_gp_wins}/8 metrics")
print(f"  XGBoost wins: {xgb_sprint_wins + xgb_gp_wins}/8 metrics")

# Determine overall winner
total_knn_wins = knn_sprint_wins + knn_gp_wins
total_xgb_wins = xgb_sprint_wins + xgb_gp_wins

print("\n" + "-"*70)
if total_xgb_wins > total_knn_wins:
    print("🎉 OVERALL WINNER: XGBoost")
    print(f"XGBoost outperforms KNN on {total_xgb_wins}/{total_knn_wins + total_xgb_wins} metrics")
elif total_knn_wins > total_xgb_wins:
    print("🎉 OVERALL WINNER: KNN")
    print(f"KNN outperforms XGBoost on {total_knn_wins}/{total_knn_wins + total_xgb_wins} metrics")
else:
    print("🤝 TIE: Both models perform equally well")

# Key insights
print("\n" + "-"*70)
print("KEY INSIGHTS")
print("-"*70)

sprint_mae_diff = xgb_metrics['sprint']['MAE'] - knn_metrics['sprint']['MAE']
gp_mae_diff = xgb_metrics['grand_prix']['MAE'] - knn_metrics['grand_prix']['MAE']

print("\nMean Absolute Error (lower is better):")
if sprint_mae_diff < 0:
    print(f"  Sprint: XGBoost improves by {abs(sprint_mae_diff):.2f} points ({abs(sprint_mae_diff/knn_metrics['sprint']['MAE']*100):.1f}%)")
else:
    print(f"  Sprint: KNN better by {abs(sprint_mae_diff):.2f} points ({abs(sprint_mae_diff/knn_metrics['sprint']['MAE']*100):.1f}%)")

if gp_mae_diff < 0:
    print(f"  Grand Prix: XGBoost improves by {abs(gp_mae_diff):.2f} points ({abs(gp_mae_diff/knn_metrics['grand_prix']['MAE']*100):.1f}%)")
else:
    print(f"  Grand Prix: KNN better by {abs(gp_mae_diff):.2f} points ({abs(gp_mae_diff/knn_metrics['grand_prix']['MAE']*100):.1f}%)")

sprint_acc_diff = xgb_metrics['sprint']['Top8_Accuracy'] - knn_metrics['sprint']['Top8_Accuracy']
gp_acc_diff = xgb_metrics['grand_prix']['Top10_Accuracy'] - knn_metrics['grand_prix']['Top10_Accuracy']

print("\nAccuracy (higher is better):")
if sprint_acc_diff > 0:
    print(f"  Sprint Top-8: XGBoost improves by {abs(sprint_acc_diff):.1f} percentage points")
else:
    print(f"  Sprint Top-8: KNN better by {abs(sprint_acc_diff):.1f} percentage points")

if gp_acc_diff > 0:
    print(f"  Grand Prix Top-10: XGBoost improves by {abs(gp_acc_diff):.1f} percentage points")
else:
    print(f"  Grand Prix Top-10: KNN better by {abs(gp_acc_diff):.1f} percentage points")

print("\n" + "="*70)

# Save comparison to CSV
sprint_comparison.to_csv('model_comparison_sprint.csv', index=False)
gp_comparison.to_csv('model_comparison_gp.csv', index=False)
print("\n✓ Saved comparison tables to:")
print("  - model_comparison_sprint.csv")
print("  - model_comparison_gp.csv")

# Save combined metrics
combined_metrics = {
    "comparison_date": "2025-12-11",
    "models_compared": ["KNN", "XGBoost"],
    "sprint_race": {
        "knn": knn_metrics['sprint'],
        "xgboost": xgb_metrics['sprint'],
        "winner_count": {
            "knn": int(knn_sprint_wins),
            "xgboost": int(xgb_sprint_wins)
        }
    },
    "grand_prix": {
        "knn": knn_metrics['grand_prix'],
        "xgboost": xgb_metrics['grand_prix'],
        "winner_count": {
            "knn": int(knn_gp_wins),
            "xgboost": int(xgb_gp_wins)
        }
    },
    "overall_winner": "XGBoost" if total_xgb_wins > total_knn_wins else "KNN" if total_knn_wins > total_xgb_wins else "Tie"
}

with open("model_comparison_summary.json", "w") as f:
    json.dump(combined_metrics, f, indent=2)

print("  - model_comparison_summary.json")
print("\n" + "="*70)
