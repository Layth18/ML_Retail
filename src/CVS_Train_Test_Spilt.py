import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load your prepared data
df = pd.read_csv('data/preparedData/prepared_data.csv')

def split_and_save_data(df, target_col='ChurnRiskCategory', test_size=0.2, random_state=42):
    # Drop identifier column
    # Use errors='ignore' in case CustomerID was already dropped in previous steps
    X = df.drop(columns=['CustomerID', target_col], errors='ignore')
    y = df[target_col]

    # Train-test split
    # Added 'stratify=y' to ensure the risk categories are balanced in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test

# 1. Execute the split
X_train, X_test, y_train, y_test = split_and_save_data(df, target_col='ChurnRiskCategory')

# 2. Define and CREATE the directory
output_dir = "data/TestTrainData/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"📁 Created directory: {output_dir}")

# 3. Save the files
X_train.to_csv(os.path.join(output_dir, "X_Train.csv"), index=False)
X_test.to_csv(os.path.join(output_dir, "X_Test.csv"), index=False)
y_train.to_csv(os.path.join(output_dir, "y_Train.csv"), index=False)
y_test.to_csv(os.path.join(output_dir, "y_Test.csv"), index=False)

print(f"💾 Success! Train/Test data saved to: {output_dir}")
print(f"   - Training rows: {len(X_train)}")
print(f"   - Testing rows:  {len(X_test)}")