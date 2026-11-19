import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from sksurv.metrics import concordance_index_ipcw, brier_score
from sksurv.util import Surv

# ---------------------------------------------------------
# 1. Synthetic Clinical / Radiomic Survival Dataset
# ---------------------------------------------------------
np.random.seed(86)
n = 600
df = pd.DataFrame({
    "age": np.random.normal(62, 10, n),
    "tumor_size": np.random.exponential(2.5, n),
    "rad_score": np.random.normal(0, 1, n),
    "group": np.random.choice(["Group_A", "Group_B"], size=n, p=[0.7, 0.3]),  # Protected attribute
})

# Simulate true log-hazard with differential baseline risk across subgroups
true_log_hr = (
    0.03 * (df["age"] - 60)
    + 0.4 * df["tumor_size"]
    + 0.6 * df["rad_score"]
    + np.where(df["group"] == "Group_B", 0.55, 0.0)
)
event_times = np.random.exponential(scale=36.0 / np.exp(true_log_hr))
censor_times = np.random.uniform(12, 60, size=n)

df["time"] = np.minimum(event_times, censor_times)
df["event"] = (event_times <= censor_times).astype(int)

# Train / Test split
train_df = df.iloc[:400].copy()
test_df = df.iloc[400:].copy()

# ---------------------------------------------------------
# 2. Fit Multivariable Cox Model (Blinded to Protected Group)
# ---------------------------------------------------------
features = ["age", "tumor_size", "rad_score"]
cph = CoxPHFitter()
cph.fit(train_df[features + ["time", "event"]], duration_col="time", event_col="event")

# Predict linear risk scores (log partial hazard) and survival curves S(t | x)
test_df["risk_score"] = cph.predict_partial_hazard(test_df[features])

# ---------------------------------------------------------
# 3. Audit A: Observed Subgroup Survival (Kaplan-Meier & Log-Rank)
# ---------------------------------------------------------
print("=== 1. Observed Subgroup Survival (Kaplan-Meier) ===")
kmf = KaplanMeierFitter()
for grp, sub_df in test_df.groupby("group"):
    kmf.fit(sub_df["time"], event_observed=sub_df["event"], label=grp)
    median_os = kmf.median_survival_time_
    s_24m = kmf.predict(24.0)
    print(f"{grp}: n={len(sub_df)}, Median OS={median_os:.1f}m, 24-mo OS={s_24m:.3f}")

lr_res = multivariate_logrank_test(test_df["time"], test_df["group"], test_df["event"])
print(f"Log-rank p-value across subgroups: {lr_res.p_value:.4e}\n")

# ---------------------------------------------------------
# 4. Audit B: Discrimination & Calibration Fairness (IPCW Metrics)
# ---------------------------------------------------------
# Convert to structured arrays required by scikit-survival
y_train_surv = Surv.from_dataframe("event", "time", train_df)
y_test_surv = Surv.from_dataframe("event", "time", test_df)

eval_time = 24.0  # Fixed clinical horizon (e.g., 24-month OS)

# Predict survival probability S(t = 24 | x) for each test patient
surv_funcs = cph.predict_survival_function(test_df[features], times=[eval_time])
test_df["surv_prob_24m"] = surv_funcs.loc[eval_time].values

print(f"=== 2. Subgroup Discrimination & Calibration at t = {eval_time} months ===")
results = []
for grp, sub_df in test_df.groupby("group"):
    idx = sub_df.index
    y_sub = Surv.from_dataframe("event", "time", sub_df)

    # Uno's IPCW C-index (accounts for group-specific censoring distributions)
    c_ipcw = concordance_index_ipcw(
        y_train_surv, y_sub, sub_df["risk_score"].values, tau=eval_time
    )[0]

    # Time-dependent IPCW Brier Score (lower is better; measures calibration + discrimination)
    _, bs = brier_score(
        y_train_surv, y_sub, sub_df["surv_prob_24m"].values.reshape(-1, 1), times=[eval_time]
    )

    # Mean predicted vs. observed KM survival at eval_time (Calibration-in-the-Large)
    kmf.fit(sub_df["time"], event_observed=sub_df["event"])
    obs_surv = kmf.predict(eval_time)
    pred_surv = sub_df["surv_prob_24m"].mean()

    results.append({
        "Group": grp,
        "N": len(sub_df),
        "Uno_C_Index": c_ipcw,
        "IPCW_Brier_24m": bs[0],
        "Observed_OS_24m": obs_surv,
        "Mean_Pred_OS_24m": pred_surv,
        "Calibration_Gap": pred_surv - obs_surv,
    })

audit_df = pd.DataFrame(results)
print(audit_df.to_string(index=False))

# ---------------------------------------------------------
# 5. Summary Fairness Disparity Gaps (Max - Min across groups)
# ---------------------------------------------------------
c_gap = audit_df["Uno_C_Index"].max() - audit_df["Uno_C_Index"].min()
brier_gap = audit_df["IPCW_Brier_24m"].max() - audit_df["IPCW_Brier_24m"].min()
cal_gap = audit_df["Calibration_Gap"].abs().max()

print("\n=== 3. Fairness Disparity Summary ===")
print(f"Uno C-Index Disparity Gap:      {c_gap:.4f} (Target: ~0.00)")
print(f"IPCW Brier Score Disparity Gap: {brier_gap:.4f} (Target: ~0.00)")
print(f"Max Subgroup Calibration Error: {cal_gap:.4f} (Target: ~0.00)")
