
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 1. LOAD THE DATASETS
# ==========================================

df = pd.read_csv("data/regression_2_2.csv")
metadata = pd.read_csv("data/team_match_metadata.csv")

print("Regression 2.2 dataset loaded successfully")
print("Original dataset shape:", df.shape)


# ==========================================
# 2. CHECK AND CLEAN COLUMN NAMES
# ==========================================

df.columns = df.columns.str.strip()
metadata.columns = metadata.columns.str.strip()

required_columns = [
    "row_id",
    "elo_difference",
    "rest_difference",
    "home_advantage",
    "competitive_share",
    "goals_for_mean",
    "opponent_goals_against_mean",
    "failed_to_score_rate",
    "opponent_clean_sheet_rate",
    "goals"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

if df["row_id"].duplicated().any():
    raise ValueError("Duplicate row_id values found in regression dataset.")

if metadata["row_id"].duplicated().any():
    raise ValueError("Duplicate row_id values found in metadata.")

if df[required_columns].isnull().any().any():
    raise ValueError("Missing values found in regression dataset.")


# ==========================================
# 3. DEFINE THE EIGHT PREDICTORS
# ==========================================

features = [
    "elo_difference",
    "rest_difference",
    "home_advantage",
    "competitive_share",
    "goals_for_mean",
    "opponent_goals_against_mean",
    "failed_to_score_rate",
    "opponent_clean_sheet_rate"
]

X = df[features]
y = df["goals"]


# ==========================================
# 4. MERGE MATCH DATES
# ==========================================

df = df.merge(
    metadata[["row_id", "date"]],
    on="row_id",
    how="left",
    validate="one_to_one"
)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

if df["date"].isnull().any():
    raise ValueError(
        "Some rows have no valid match date. Check the metadata file."
    )

# Extract match ID from row IDs such as M001_A and M001_B.
df["match_id"] = df["row_id"].str.rsplit("_", n=1).str[0]

# Check that every match has exactly two team rows.
rows_per_match = df.groupby("match_id").size()

if not (rows_per_match == 2).all():
    raise ValueError("Some matches do not have exactly two team rows.")

# Ensure each match has one consistent date.
dates_per_match = df.groupby("match_id")["date"].nunique()

if not (dates_per_match == 1).all():
    raise ValueError("Inconsistent dates found for a match.")


# 5. SPLIT MATCHES CHRONOLOGICALLY
# ==========================================

# Create one record per match.
match_dates = (
    df[["match_id", "date"]]
    .drop_duplicates()
    .sort_values(["date", "match_id"])
    .reset_index(drop=True)
)

# Count matches played on each date.
matches_per_date = (
    match_dates.groupby("date")["match_id"]
    .count()
    .sort_index()
)

# Find a date cutoff close to 80% of all matches.
target_training_matches = int(len(match_dates) * 0.8)

cumulative_matches = matches_per_date.cumsum()

cutoff_position = (
    cumulative_matches - target_training_matches
).abs().argmin()

cutoff_date = matches_per_date.index[cutoff_position]

# Keep every match on a given date in the same split.
train_mask = df["date"] <= cutoff_date
test_mask = df["date"] > cutoff_date

# Create predictors and target AFTER merging metadata.
X = df[features]
y = df["goals"]

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("\n--- Dataset Split ---")
print("Total matches:", df["match_id"].nunique())
print("Training matches:", df.loc[train_mask, "match_id"].nunique())
print("Testing matches:", df.loc[test_mask, "match_id"].nunique())
print("Training team rows:", len(X_train))
print("Testing team rows:", len(X_test))
print("Latest training date:", df.loc[train_mask, "date"].max().date())
print("Earliest testing date:", df.loc[test_mask, "date"].min().date())

# Verify that the split is chronological.
assert df.loc[train_mask, "date"].max() < df.loc[test_mask, "date"].min()

# Verify that every match stays entirely within one split.
train_match_ids = set(df.loc[train_mask, "match_id"])
test_match_ids = set(df.loc[test_mask, "match_id"])

assert train_match_ids.isdisjoint(test_match_ids)

# 6. TRAIN THE LINEAR REGRESSION MODEL
# ==========================================

model = LinearRegression()
model.fit(X_train, y_train)


# ==========================================
# 7. PREDICT GOALS
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# 8. EVALUATE THE MODEL
# ==========================================

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("\n--- Regression 2.2 Results ---")
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R²:", round(r2, 3))


# ==========================================
# 9. COMPARE WITH A BASELINE
# ==========================================

# Baseline predicts the mean goals from training data
baseline_predictions = np.full(
    len(y_test),
    y_train.mean()
)

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_predictions)
)

print("\n--- Baseline Results ---")
print("Baseline MAE:", round(baseline_mae, 3))
print("Baseline RMSE:", round(baseline_rmse, 3))

print("\n--- Model vs Baseline ---")
print(
    "MAE improvement:",
    round(baseline_mae - mae, 3)
)

print(
    "RMSE improvement:",
    round(baseline_rmse - rmse, 3)
)


# ==========================================
# 10. DISPLAY MODEL COEFFICIENTS
# ==========================================

coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

print("\n--- Model Coefficients ---")
print(coefficients.to_string(index=False))
print("Intercept:", round(model.intercept_, 3))


# ==========================================
# 11. SAVE TEST PREDICTIONS
# ==========================================

results = df.loc[
    test_mask,
    ["row_id", "match_id", "date", "goals"]
].copy()

results["predicted_goals"] = predictions
results["prediction_error"] = (
    results["goals"] - results["predicted_goals"]
).abs()

results.to_csv(
    "data/regression_2_2_predictions.csv",
    index=False
)

print("\nPredictions saved to:")
print("data/regression_2_2_predictions.csv")

#to generate the images
# ==========================================
# 12. SECTION 6.1 — STATISTICAL SIGNIFICANCE
# AND 95% CONFIDENCE INTERVAL CHART
# ==========================================

import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

# Create folder for report images
output_folder = Path("report_images")
output_folder.mkdir(exist_ok=True)

# Add intercept for statistical inference
X_train_sm = sm.add_constant(X_train, has_constant="add")

# Use match-clustered standard errors because
# each match contributes two team observations.
training_match_ids = df.loc[train_mask, "match_id"]

ols_results = sm.OLS(y_train, X_train_sm).fit(
    cov_type="cluster",
    cov_kwds={"groups": training_match_ids}
)

# Get coefficients, confidence intervals, and p-values
confidence = ols_results.conf_int(alpha=0.05)

inference_table = pd.DataFrame({
    "Feature": features,
    "Coefficient": ols_results.params[features].values,
    "CI Lower": confidence.loc[features, 0].values,
    "CI Upper": confidence.loc[features, 1].values,
    "P Value": ols_results.pvalues[features].values
})

print("\n--- Section 6.1 Statistical Results ---")
print(inference_table.to_string(index=False))

# Save the numerical results
inference_table.to_csv(
    output_folder / "section_6_1_statistics.csv",
    index=False
)

# Prepare confidence interval chart
plot_data = inference_table.sort_values("Coefficient")

fig, ax = plt.subplots(figsize=(11, 7))

fig.patch.set_facecolor("#f5f7fb")
ax.set_facecolor("white")

positions = np.arange(len(plot_data))

lower_error = (
    plot_data["Coefficient"] - plot_data["CI Lower"]
)

upper_error = (
    plot_data["CI Upper"] - plot_data["Coefficient"]
)

ax.errorbar(
    plot_data["Coefficient"],
    positions,
    xerr=[lower_error, upper_error],
    fmt="o",
    color="#2563eb",
    ecolor="#64748b",
    capsize=5,
    markersize=8,
    linewidth=2
)

ax.axvline(
    0,
    color="#dc2626",
    linestyle="--",
    linewidth=1.5,
    label="Zero effect"
)

ax.set_yticks(positions)
ax.set_yticklabels(plot_data["Feature"])

ax.set_xlabel("Regression Coefficient (95% Confidence Interval)")
ax.set_title(
    "Regression 2.2: Coefficient Confidence Intervals",
    fontsize=16,
    fontweight="bold",
    pad=18
)

ax.grid(axis="x", alpha=0.2)
ax.legend()

plt.tight_layout()

image_path = output_folder / "section_6_1_confidence_intervals.png"

plt.savefig(
    image_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSection 6.1 image saved to:", image_path)

#for 6.2
# ==========================================
# 13. SECTION 6.2 — REGRESSION DIAGNOSTICS
# ==========================================

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats
from pathlib import Path

output_folder = Path("report_images")
output_folder.mkdir(exist_ok=True)

# Use training data for regression diagnostics
X_diagnostic = sm.add_constant(X_train, has_constant="add")
diagnostic_model = sm.OLS(y_train, X_diagnostic).fit()

fitted_values = diagnostic_model.fittedvalues
residuals = diagnostic_model.resid

# Calculate influence statistics
influence = diagnostic_model.get_influence()
standardized_residuals = influence.resid_studentized_internal
cooks_distance = influence.cooks_distance[0]

# Create 2x2 diagnostic dashboard
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# ------------------------------------------
# Plot 1: Residuals vs Fitted Values
# ------------------------------------------

axes[0, 0].scatter(
    fitted_values,
    residuals,
    alpha=0.7,
    color="royalblue"
)

axes[0, 0].axhline(
    0,
    color="red",
    linestyle="--"
)

axes[0, 0].set_title("Residuals vs Fitted Values")
axes[0, 0].set_xlabel("Fitted Goals")
axes[0, 0].set_ylabel("Residuals")

# ------------------------------------------
# Plot 2: Normal Q-Q Plot
# ------------------------------------------

sm.qqplot(
    residuals,
    line="45",
    ax=axes[0, 1],
    markerfacecolor="royalblue",
    markeredgecolor="royalblue"
)

axes[0, 1].set_title("Normal Q-Q Plot")

# ------------------------------------------
# Plot 3: Residual Distribution
# ------------------------------------------

axes[1, 0].hist(
    residuals,
    bins=15,
    color="steelblue",
    edgecolor="white"
)

axes[1, 0].set_title("Residual Distribution")
axes[1, 0].set_xlabel("Residual")
axes[1, 0].set_ylabel("Frequency")

# ------------------------------------------
# Plot 4: Cook's Distance
# ------------------------------------------

axes[1, 1].stem(
    np.arange(len(cooks_distance)),
    cooks_distance,
    linefmt="steelblue",
    markerfmt="bo",
    basefmt=" "
)

threshold = 4 / len(X_train)

axes[1, 1].axhline(
    threshold,
    color="red",
    linestyle="--",
    label=f"Threshold = {threshold:.3f}"
)

axes[1, 1].set_title("Cook's Distance")
axes[1, 1].set_xlabel("Training Observation")
axes[1, 1].set_ylabel("Cook's Distance")
axes[1, 1].legend()

# ------------------------------------------
# Save image
# ------------------------------------------

fig.suptitle(
    "Regression 2.2 — Model Diagnostic Analysis",
    fontsize=17,
    fontweight="bold"
)

plt.tight_layout()

image_path = (
    output_folder / "section_6_2_diagnostics.png"
)

plt.savefig(
    image_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSection 6.2 image saved to:", image_path)

# ------------------------------------------
# Additional statistical tests
# ------------------------------------------

from statsmodels.stats.diagnostic import het_breuschpagan

shapiro_stat, shapiro_p = stats.shapiro(residuals)

bp_stat, bp_p, _, _ = het_breuschpagan(
    residuals,
    X_diagnostic
)

print("\n--- Section 6.2 Diagnostic Tests ---")
print("Shapiro-Wilk p-value:", round(shapiro_p, 4))
print("Breusch-Pagan p-value:", round(bp_p, 4))
print(
    "Observations above Cook's threshold:",
    int(np.sum(cooks_distance > threshold))
)
# ==========================================
# 14. SECTION 6.3 — BASELINE COMPARISON
# ==========================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

output_folder = Path("report_images")
output_folder.mkdir(exist_ok=True)

# Full eight-predictor model (already trained)
full_predictions = model.predict(X_test)

# Elo-only model
elo_model = LinearRegression()
elo_model.fit(X_train[["elo_difference"]], y_train)

elo_predictions = elo_model.predict(
    X_test[["elo_difference"]]
)

# Mean baseline
mean_predictions = np.full(
    len(y_test),
    y_train.mean()
)

def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    }

comparison = pd.DataFrame({
    "Full 8-Predictor": calculate_metrics(
        y_test, full_predictions
    ),
    "Elo-Only": calculate_metrics(
        y_test, elo_predictions
    ),
    "Mean Baseline": calculate_metrics(
        y_test, mean_predictions
    )
}).T

print("\n--- Section 6.3 Baseline Comparison ---")
print(comparison.round(3))

comparison.to_csv(
    output_folder / "section_6_3_comparison.csv"
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
    bars = ax.bar(
        comparison.index,
        comparison[metric]
    )

    ax.set_title(metric, fontweight="bold")
    ax.tick_params(axis="x", rotation=25)
    ax.axhline(0, color="black", linewidth=0.8)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5 if height >= 0 else -14),
            textcoords="offset points",
            ha="center"
        )

fig.suptitle(
    "Regression 2.2 — Baseline Model Comparison",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()
plt.savefig(
    output_folder / "section_6_3_baseline_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Section 6.3 image generated.")


# ==========================================
# 15. SECTION 6.4 — EXPANDING-WINDOW
# CROSS-VALIDATION
# ==========================================

# Use only training matches to avoid test leakage
training_df = (
    df.loc[train_mask]
    .sort_values(["date", "match_id", "row_id"])
    .copy()
)

# Keep all matches from the same date together
unique_dates = sorted(training_df["date"].unique())

# Three chronological folds:
# 50%, 65%, and 80% of training dates
fold_proportions = [0.50, 0.65, 0.80]

cv_results = []

for fold_number, proportion in enumerate(
    fold_proportions, start=1
):
    split_index = int(len(unique_dates) * proportion)

    # Validation ends at the next fold boundary
    if fold_number < len(fold_proportions):
        end_index = int(
            len(unique_dates)
            * fold_proportions[fold_number]
        )
    else:
        end_index = len(unique_dates)

    train_dates = unique_dates[:split_index]
    validation_dates = unique_dates[
        split_index:end_index
    ]

    fold_train = training_df[
        training_df["date"].isin(train_dates)
    ]

    fold_validation = training_df[
        training_df["date"].isin(validation_dates)
    ]

    if fold_train.empty or fold_validation.empty:
        continue

    cv_model = LinearRegression()

    cv_model.fit(
        fold_train[features],
        fold_train["goals"]
    )

    cv_predictions = cv_model.predict(
        fold_validation[features]
    )

    cv_baseline = np.full(
        len(fold_validation),
        fold_train["goals"].mean()
    )

    cv_results.append({
        "Fold": fold_number,
        "Train Matches": fold_train["match_id"].nunique(),
        "Validation Matches": fold_validation["match_id"].nunique(),
        "Model MAE": mean_absolute_error(
            fold_validation["goals"],
            cv_predictions
        ),
        "Baseline MAE": mean_absolute_error(
            fold_validation["goals"],
            cv_baseline
        ),
        "R2": r2_score(
            fold_validation["goals"],
            cv_predictions
        )
    })

cv_table = pd.DataFrame(cv_results)

print("\n--- Section 6.4 Cross-Validation ---")
print(cv_table.round(3).to_string(index=False))

cv_table.to_csv(
    output_folder / "section_6_4_cross_validation.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(9, 5))

positions = np.arange(len(cv_table))
width = 0.35

ax.bar(
    positions - width / 2,
    cv_table["Model MAE"],
    width,
    label="Full Model"
)

ax.bar(
    positions + width / 2,
    cv_table["Baseline MAE"],
    width,
    label="Mean Baseline"
)

ax.set_xticks(positions)
ax.set_xticklabels(
    [f"Fold {x}" for x in cv_table["Fold"]]
)

ax.set_ylabel("Mean Absolute Error")
ax.set_title(
    "Regression 2.2 — Expanding-Window Validation",
    fontweight="bold"
)

ax.legend()
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(
    output_folder / "section_6_4_cross_validation.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Section 6.4 image generated.")


# ==========================================
# 16. SECTION 6.5 — MATCH-LEVEL BOOTSTRAP
# ==========================================

# Resample whole test matches, not individual rows,
# because each match contributes two observations.

test_data = df.loc[test_mask].copy()

test_data["full_prediction"] = full_predictions
test_data["baseline_prediction"] = mean_predictions

match_ids = test_data["match_id"].unique()

rng = np.random.default_rng(42)

bootstrap_mae = []
bootstrap_improvement = []

for _ in range(3000):

    sampled_matches = rng.choice(
        match_ids,
        size=len(match_ids),
        replace=True
    )

    sampled = pd.concat(
        [
            test_data[
                test_data["match_id"] == match_id
            ]
            for match_id in sampled_matches
        ],
        ignore_index=True
    )

    actual = sampled["goals"]

    model_mae = mean_absolute_error(
        actual,
        sampled["full_prediction"]
    )

    baseline_mae_sample = mean_absolute_error(
        actual,
        sampled["baseline_prediction"]
    )

    bootstrap_mae.append(model_mae)

    bootstrap_improvement.append(
        baseline_mae_sample - model_mae
    )

mae_interval = np.percentile(
    bootstrap_mae, [2.5, 97.5]
)

improvement_interval = np.percentile(
    bootstrap_improvement, [2.5, 97.5]
)

print("\n--- Section 6.5 Bootstrap Results ---")
print("MAE 95% CI:", mae_interval)
print(
    "Improvement 95% CI:",
    improvement_interval
)

bootstrap_summary = pd.DataFrame({
    "Metric": [
        "Model MAE",
        "Baseline MAE minus Model MAE"
    ],
    "Lower 95% CI": [
        mae_interval[0],
        improvement_interval[0]
    ],
    "Upper 95% CI": [
        mae_interval[1],
        improvement_interval[1]
    ]
})

bootstrap_summary.to_csv(
    output_folder / "section_6_5_bootstrap_summary.csv",
    index=False
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(
    bootstrap_mae,
    bins=30,
    edgecolor="white"
)

axes[0].axvline(
    mae_interval[0],
    linestyle="--",
    color="red"
)

axes[0].axvline(
    mae_interval[1],
    linestyle="--",
    color="red"
)

axes[0].set_title("Bootstrap Distribution of Model MAE")
axes[0].set_xlabel("MAE")

axes[1].hist(
    bootstrap_improvement,
    bins=30,
    edgecolor="white"
)

axes[1].axvline(
    0,
    color="black",
    linestyle=":"
)

axes[1].axvline(
    improvement_interval[0],
    color="red",
    linestyle="--"
)

axes[1].axvline(
    improvement_interval[1],
    color="red",
    linestyle="--"
)

axes[1].set_title(
    "Bootstrap Improvement Over Mean Baseline"
)
axes[1].set_xlabel("Baseline MAE - Model MAE")

fig.suptitle(
    "Regression 2.2 — Bootstrap Uncertainty",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()
plt.savefig(
    output_folder / "section_6_5_bootstrap.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Section 6.5 image generated.")