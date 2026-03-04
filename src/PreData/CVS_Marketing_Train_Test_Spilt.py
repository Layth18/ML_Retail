import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# ─────────────────────────────────────────────
# 1. Setup paths
# ─────────────────────────────────────────────
raw_base_path = 'data/preparedData/X_unscaled.csv'
output_dir = 'data/MarketingTimelineData/'
os.makedirs(output_dir, exist_ok=True)

# ─────────────────────────────────────────────
# 2. Load raw data and persona model
# ─────────────────────────────────────────────
df_unscaled = pd.read_csv(raw_base_path)
persona_model = joblib.load('models/persona_classifier.pkl')
main_scaler = joblib.load('models/main_scaler.pkl')

# ─────────────────────────────────────────────
# 3. Generate Persona labels
# ─────────────────────────────────────────────
X_scaled_temp = pd.DataFrame(
    main_scaler.transform(df_unscaled),
    columns=df_unscaled.columns
)

df_unscaled['Persona'] = persona_model.predict(X_scaled_temp)

# ─────────────────────────────────────────────
# 4. FIX REGION IMBALANCE (NEW STEP)
# ─────────────────────────────────────────────
min_customers = 50
region_counts = df_unscaled['Region'].value_counts()

df_unscaled['RegionGrouped'] = df_unscaled['Region'].apply(
    lambda x: x if region_counts[x] >= min_customers else 'Other'
)

print("📊 Region distribution after grouping:")
print(df_unscaled['RegionGrouped'].value_counts())

# ─────────────────────────────────────────────
# 5. Feature Engineering
# ─────────────────────────────────────────────
df_unscaled['TargetSpendingPerSeason'] = (
    (df_unscaled['MonetaryTotal'] / df_unscaled['Frequency'].replace(0, np.nan))
    * (df_unscaled['CustomerTenureDays'] / 365)
)

df_unscaled['TargetSpendingPerSeason'] = (
    df_unscaled['TargetSpendingPerSeason']
    .replace([np.inf, -np.inf], 0)
    .fillna(0)
)

features_to_keep = [
    'Recency',
    'Frequency',
    'CustomerTenureDays',
    'WeekendPurchaseRatio',
    'FavoriteSeason',
    'RegionGrouped',   # ← use grouped version
    'Persona'
]

marketing_df = df_unscaled[features_to_keep + ['TargetSpendingPerSeason']].copy()

# ─────────────────────────────────────────────
# 6. One-hot encode categorical columns (KEEP ALL CATEGORIES)
# ─────────────────────────────────────────────
marketing_df = pd.get_dummies(
    marketing_df,
    columns=['FavoriteSeason', 'RegionGrouped', 'Persona'],
    prefix=['Season', 'Reg', 'Pers'],
    drop_first=False   # KEEP all categories so Season_0, Persona_0 exist
)
# ─────────────────────────────────────────────
# 7. Split into train/test
# ─────────────────────────────────────────────
X = marketing_df.drop('TargetSpendingPerSeason', axis=1)
y = marketing_df['TargetSpendingPerSeason']

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# 8. Scale numeric features only
# ─────────────────────────────────────────────
numeric_cols = [
    'Recency',
    'Frequency',
    'CustomerTenureDays',
    'WeekendPurchaseRatio'
]

marketing_scaler = StandardScaler()

X_train_m_scaled = X_train_m.copy()
X_test_m_scaled = X_test_m.copy()

X_train_m_scaled[numeric_cols] = marketing_scaler.fit_transform(
    X_train_m[numeric_cols]
)

X_test_m_scaled[numeric_cols] = marketing_scaler.transform(
    X_test_m[numeric_cols]
)

# ─────────────────────────────────────────────
# 9. Save datasets
# ─────────────────────────────────────────────
X_train_m_scaled.to_csv(os.path.join(output_dir, "X_Train_Marketing.csv"), index=False)
X_test_m_scaled.to_csv(os.path.join(output_dir, "X_Test_Marketing.csv"), index=False)
y_train_m.to_csv(os.path.join(output_dir, "y_Train_Marketing.csv"), index=False)
y_test_m.to_csv(os.path.join(output_dir, "y_Test_Marketing.csv"), index=False)

# Save full dataset properly ordered
full_scaled_df = pd.concat([X_train_m_scaled, X_test_m_scaled], axis=0)
full_y = pd.concat([y_train_m, y_test_m], axis=0)

full_scaled_df['TargetSpendingPerSeason'] = full_y.values
full_scaled_df.to_csv(os.path.join(output_dir, "XY_Full_Marketing.csv"), index=False)

joblib.dump(marketing_scaler, 'models/marketing_timeline_scaler.pkl')

print("✅ Marketing Timeline datasets created (split & full)")
print(f"📊 Feature count: {X_train_m_scaled.shape[1]} (including one-hot columns)")