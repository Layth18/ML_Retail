import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import sys

df = pd.read_csv('data/raw_data.csv')

def clean_data_with_reports(df, var_threshold=0.95, corr_threshold=0.8, outlier_contamination=0.05):
    """
    Cleans the dataframe with real-time feedback after each step.
    """
    # Helper to calculate memory usage in KB
    get_mem = lambda d: d.memory_usage(deep=True).sum() / 1024
    
    initial_rows, initial_cols = df.shape
    initial_mem = get_mem(df)
    
    print(f"🚀 Starting Cleaning...")
    print(f"Initial State: {initial_rows} rows, {initial_cols} columns | Size: {initial_mem:.2f} KB\n")
    print("-" * 50)

    # 1. Remove Duplicate Rows
    rows_before = df.shape[0]
    df = df.drop_duplicates()
    removed = rows_before - df.shape[0]
    print(f"✅ STEP 1: Duplicate Rows")
    print(f"   Removed: {removed} rows | Current Size: {get_mem(df):.2f} KB")
    print("-" * 50)

    # 2. Remove Low Variance (Overrepresentation)
    cols_before = df.shape[1]
    cols_to_drop_var = [col for col in df.columns if df[col].value_counts(normalize=True).iloc[0] >= var_threshold]
    df = df.drop(columns=cols_to_drop_var)
    print(f"✅ STEP 2: Low Variance Columns")
    print(f"   Dropped: {len(cols_to_drop_var)} columns ({', '.join(cols_to_drop_var) if cols_to_drop_var else 'None'})")
    print(f"   Current Size: {get_mem(df):.2f} KB")
    print("-" * 50)

    # 3. Remove High Correlation
    cols_before = df.shape[1]
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    cols_to_drop_corr = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    df = df.drop(columns=cols_to_drop_corr)
    print(f"✅ STEP 3: High Correlation")
    print(f"   Dropped: {len(cols_to_drop_corr)} columns ({', '.join(cols_to_drop_corr) if cols_to_drop_corr else 'None'})")
    print(f"   Current Size: {get_mem(df):.2f} KB")
    print("-" * 50)

    # 4. Remove Extreme Points (Isolation Forest)
    rows_before = df.shape[0]
    # Filling NaNs with median just for the outlier detection calculation
    temp_numeric = df.select_dtypes(include=[np.number]).fillna(df.median(numeric_only=True))
    iso = IsolationForest(contamination=outlier_contamination, random_state=42)
    outlier_preds = iso.fit_predict(temp_numeric)
    
    df = df[outlier_preds == 1]
    removed = rows_before - df.shape[0]
    print(f"✅ STEP 4: Extreme Points (Outliers)")
    print(f"   Removed: {removed} rows (based on {outlier_contamination*100}% contamination)")
    print(f"   Current Size: {get_mem(df):.2f} KB")
    print("-" * 50)

    # FINAL SUMMARY
    final_mem = get_mem(df)
    reduction = ((initial_mem - final_mem) / initial_mem * 100) if initial_mem > 0 else 0
    
    print(f"✨ FINAL CLEANING SUMMARY")
    print(f"Final Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Total Memory Reduction: {reduction:.2f}% slimmer")
    
    return df

# Execute
df_cleaned = clean_data_with_reports(df)
output_path = "data/preparedData/cleaned_data.csv"
df_cleaned.to_csv(output_path, index=False)

print(f"\n💾 Cleaned data saved to: {output_path}")