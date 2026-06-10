import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

# Load final dataset
df = pd.read_csv("data/final_training_dataset.csv")

print("Dataset shape:", df.shape)

# Features and target
features = [col for col in df.columns if col not in [
    "energy_kwh",
    "log_energy_kwh"
]]

X = df[features]
y = df["log_energy_kwh"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# XGBoost model
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
pred_log = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, pred_log)

rmse = np.sqrt(mean_squared_error(y_test, pred_log))

r2 = r2_score(y_test, pred_log)

print("\nMODEL PERFORMANCE")
print("------------------")

print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

# Save model
joblib.dump(
    model,
    "models/xgboost_energy_forecaster.pkl"
)

print("\nSaved model:")
print("models/xgboost_energy_forecaster.pkl")