import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ─────────────────────────────────────────────
# 1. Load the NEW processed dataset
# ─────────────────────────────────────────────
data_path = 'data/MarketingTimelineData/XY_Full_Marketing.csv'
df = pd.read_csv(data_path)

os.makedirs('models', exist_ok=True)

# ─────────────────────────────────────────────
# 2. Separate features and target
# ─────────────────────────────────────────────
X = df.drop('TargetSpendingPerSeason', axis=1)
y = df['TargetSpendingPerSeason']

# ─────────────────────────────────────────────
# 3. Train/Test Split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# 4. Scale ONLY numeric columns
# ─────────────────────────────────────────────
numeric_cols = [
    'Recency',
    'Frequency',
    'CustomerTenureDays',
    'WeekendPurchaseRatio'
]

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ─────────────────────────────────────────────
# 5. Train RandomForestRegressor
# ─────────────────────────────────────────────
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=15,
    min_samples_leaf=6,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

# ─────────────────────────────────────────────
# 6. Evaluate
# ─────────────────────────────────────────────
y_pred = rf_model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n✅ RandomForest Regressor trained on NEW dataset")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test R²: {r2:.4f}")

# ─────────────────────────────────────────────
# 7. Save Model + Scaler ONLY
# ─────────────────────────────────────────────
joblib.dump(rf_model, 'models/marketing_timeline_model.pkl')
joblib.dump(scaler, 'models/marketing_timeline_scaler.pkl')

print("💾 Model and scaler saved successfully.")