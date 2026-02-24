import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Added for scaling
import os

# Load your prepared data
df = pd.read_csv('data/preparedData/prepared_data.csv')

def split_and_save_data(df, target_col='ChurnRiskCategory', test_size=0.2, random_state=42):
    # 1. Separate Features and Target
    X = df.drop(columns=['CustomerID', target_col], errors='ignore')
    y = df[target_col]

    # 2. Train-test split (with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # --- STEP 8: FEATURE SCALING (X ONLY) ---
    # We only scale numeric columns to avoid breaking categorical one-hot encodings
    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    
    scaler = StandardScaler()
    
    # Fit ONLY on training data to prevent leakage
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    
    # Transform test data using the training fit
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    print(f"⚖️ Feature Scaling applied to {len(numeric_cols)} numeric columns.")
    # ----------------------------------------

    return X_train, X_test, y_train, y_test

# Execute the split and scale
X_train, X_test, y_train, y_test = split_and_save_data(df)

# Define and CREATE the directory
output_dir = "data/TestTrainData/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the files
X_train.to_csv(os.path.join(output_dir, "X_Train.csv"), index=False)
X_test.to_csv(os.path.join(output_dir, "X_Test.csv"), index=False)
y_train.to_csv(os.path.join(output_dir, "y_Train.csv"), index=False)
y_test.to_csv(os.path.join(output_dir, "y_Test.csv"), index=False)

print(f"💾 Success! Scaled X and untouched y saved to: {output_dir}")