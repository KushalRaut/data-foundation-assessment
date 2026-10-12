
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
