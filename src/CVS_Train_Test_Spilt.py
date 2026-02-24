import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Load your prepared data
df = pd.read_csv('data/preparedData/prepared_data.csv')

def split_and_save_data(df, target_col='ChurnRiskCategory', test_size=0.2, random_state=42):
    # --- TOP 10 FEATURES ---
    elite_features = [
    'Recency', 
    'FirstPurchaseDaysAgo', 
    'FavoriteSeason', 
    'CustomerTenureDays', 
    'PreferredMonth_cos', 
    'CustomerType', 
    'RFMSegment', 
    'WeekendPurchaseRatio', 
    'Region', 
    'MonetaryTotal'
]
    
    # 1. Separate Features and Target
    available_features = [f for f in elite_features if f in df.columns]
    
    # FIX: Convert to float immediately to prevent LossySetitemError/TypeError during scaling
    X = df[available_features].copy().astype(float)
    y = df[target_col]

    print(f"🎯 Selected Top {len(available_features)} features. Converting to float for scaling...")

    # 2. Train-test split (with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # --- STEP 3: FEATURE SCALING (X ONLY) ---
    scaler = StandardScaler()
    
    # Scaling converts the data into a NumPy array of floats, 
    # which now fits perfectly into our float-typed DataFrames.
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns, 
        index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X_test.columns, 
        index=X_test.index
    )
    
    print(f"⚖️ Feature Scaling complete (Mean ≈ 0, Std ≈ 1)")

    return X_train, X_test, y_train, y_test

# Execute the split and scale
X_train, X_test, y_train, y_test = split_and_save_data(df)

# Define and CREATE the directory
output_dir = "data/TestTrainData/"
os.makedirs(output_dir, exist_ok=True)

# Save the files
X_train.to_csv(os.path.join(output_dir, "X_Train.csv"), index=False)
X_test.to_csv(os.path.join(output_dir, "X_Test.csv"), index=False)
y_train.to_csv(os.path.join(output_dir, "y_Train.csv"), index=False)
y_test.to_csv(os.path.join(output_dir, "y_Test.csv"), index=False)

print(f"💾 Success! Files saved to: {output_dir}")