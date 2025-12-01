import pandas as pd
import numpy as np
import os

# Change to project root if running from scripts directory
if os.path.basename(os.getcwd()) == "scripts":
    os.chdir("..")

# 1) Load your predictions
pred = pd.read_csv("remaining_races_predictions.csv")

# 2) Build ACTUALS for Sprint (R23) and GP (R24)

# Sprint points (top 8)
sprint_order = [
    ("Piastri",8), ("Russell",7), ("Norris",6), ("Verstappen",5),
    ("Tsunoda",4), ("Antonelli",3), ("Alonso",2), ("Sainz",1)
]
act_sprint = pd.DataFrame([
    {"round":23, "driverKey":n, "y_true_points":pts, "y_true_top8": True}
    for n,pts in sprint_order
])
# Non-scorers get 0; will be added when merging with preds

# GP points (top 10)
gp_order = [
    ("Verstappen",25), ("Piastri",18), ("Sainz",15), ("Norris",12),
    ("Antonelli",10), ("Russell",8), ("Alonso",6), ("Leclerc",4),
    ("Lawson",2), ("Tsunoda",1)
]
act_gp = pd.DataFrame([
    {"round":24, "driverKey":n, "y_true_points":pts, "y_true_top10": True}
    for n,pts in gp_order
])

# 3) Make a driver key in your preds to join on (adjust if you use driverId)
def norm_name(s):
    return str(s).split()[-1].lower()  # naive: use surname

# Use the correct column name from remaining_races_predictions.csv
if "driver_name" in pred.columns:
    pred["driverKey"] = pred["driver_name"].apply(norm_name)
elif "surname" in pred.columns:
    pred["driverKey"] = pred["surname"].apply(norm_name)
else:
    # Fallback: try any column containing driver info
    col_name = [c for c in pred.columns if "driver" in c.lower()][0]
    pred["driverKey"] = pred[col_name].apply(norm_name)

# Rename predicted_points to y_pred_points for consistency
if "predicted_points" in pred.columns:
    pred["y_pred_points"] = pred["predicted_points"]
elif "y_pred_points" not in pred.columns:
    raise ValueError("Could not find predicted points column")

# Normalize our actuals keys similarly
act_sprint["driverKey"] = act_sprint["driverKey"].apply(lambda s: s.lower())
act_gp["driverKey"] = act_gp["driverKey"].apply(lambda s: s.lower())

# 4) Attach actuals; fill non-scorers with 0 points / False
def eval_round(df_pred, act_df, top_flag_col):
    df = df_pred.merge(act_df[["round","driverKey","y_true_points", top_flag_col]],
                       on=["round","driverKey"], how="left")
    df["y_true_points"] = df["y_true_points"].fillna(0.0)
    if top_flag_col not in df:
        df[top_flag_col] = df[top_flag_col].fillna(False)
    # If no explicit predicted top flag, infer from predicted points > 0
    if "y_pred_top10" not in df.columns and top_flag_col=="y_true_top10":
        df["y_pred_top10"] = df["y_pred_points"] > 0
    if "y_pred_top8" not in df.columns and top_flag_col=="y_true_top8":
        df["y_pred_top8"] = df["y_pred_points"] > 0
    return df

pred_sprint = eval_round(pred[pred["round"]==23].copy(), act_sprint, "y_true_top8")
pred_gp     = eval_round(pred[pred["round"]==24].copy(), act_gp, "y_true_top10")

# 5) Metrics
def metrics(df, top_flag_true, top_flag_pred):
    err = df["y_pred_points"] - df["y_true_points"]
    mae = err.abs().mean()
    mse = (err**2).mean()
    rmse = np.sqrt(mse)
    acc = (df[top_flag_true] == df[top_flag_pred]).mean()
    # Biggest misses
    big_abs = df.loc[err.abs().sort_values(ascending=False).index, 
                     ["driverKey","y_pred_points","y_true_points"]].head(5)
    flips = df[(df[top_flag_true] != df[top_flag_pred])]
    return mae, mse, rmse, acc, big_abs, flips

mae_s, mse_s, rmse_s, acc_s, big_s, flips_s = metrics(pred_sprint, "y_true_top8", "y_pred_top8")
mae_g, mse_g, rmse_g, acc_g, big_g, flips_g = metrics(pred_gp, "y_true_top10", "y_pred_top10")

print("SPRINT (R23) — MAE:", mae_s, "MSE:", mse_s, "RMSE:", rmse_s, "Top-8 Acc:", acc_s)
print(big_s)
print("Top-8 misses:\n", flips_s[["driverKey","y_pred_points","y_true_points"]].head(10))

print("\nGRAND PRIX (R24) — MAE:", mae_g, "MSE:", mse_g, "RMSE:", rmse_g, "Top-10 Acc:", acc_g)
print(big_g)
print("Top-10 misses:\n", flips_g[["driverKey","y_pred_points","y_true_points"]].head(10))
