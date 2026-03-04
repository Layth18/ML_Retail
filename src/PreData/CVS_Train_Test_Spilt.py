import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load data
df = pd.read_csv('data/preparedData/prepared_data.csv')

def split_and_save_data(df):
    # THE ELITE 3
    elite_features = [
        'Recency', 'Frequency', 'CustomerTenureDays',
    ]
    
    # Check for missing columns
    missing = [f for f in elite_features if f not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing columns in CSV: {missing}")

    # Prepare X and y
    X = df[elite_features].copy().astype(float)
    y = df['ChurnRiskCategory']

    # --- SAVE X_unscaled IN data/preparedData/ ---
    prep_dir = "data/preparedData/"
    os.makedirs(prep_dir, exist_ok=True)
    X.to_csv(os.path.join(prep_dir, "X_unscaled.csv"), index=False)

    # Set up output directory for splits
    output_dir = "data/TestTrainData/"
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. SAVE UNSCALED & UNSPLIT FULL DATASET ---

    # --- 2. SPLIT DATA ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    
    # --- 4. SCALING ---
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # --- 5. SAVE SCALED SPLITS & LABELS ---
    X_train_scaled.to_csv(os.path.join(output_dir, "X_Train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(output_dir, "X_Test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_Train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_Test.csv"), index=False)
    
    # --- 6. SAVE SCALER ---
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/main_scaler.pkl')
    
    print("✅ All files saved successfully:")
    print(f"   - Main Unscaled X: {prep_dir}X_unscaled.csv")
    print(f"   - Full Dataset: {output_dir}full_unscaled_data.csv")
    print(f"   - Train/Test (Scaled & Unscaled): {output_dir}")
    print(f"   - Scaler: models/main_scaler.pkl")

split_and_save_data(df)
