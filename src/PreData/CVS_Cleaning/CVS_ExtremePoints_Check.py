from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np


df = pd.read_csv('data/raw_data.csv')

def check_extreme_cases(df, contamination=0.05):
    # Isolation Forest requires numeric data with no NaNs
    # Ensure you have filled missing values before running this
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    
    # Initialize the model
    # 'contamination' is the % of data you expect to be outliers
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    
    # Fit and predict
    outliers = iso_forest.fit_predict(numeric_df)
    
    # Count the -1 values
    num_outliers = (outliers == -1).sum()
    
    print(f"--- Extreme Cases (Isolation Forest) ---")
    print(f"Number of extreme cases found: {num_outliers}")
    print(f"Percentage of dataset flagged: {contamination * 100}%")
    
    return num_outliers

# Usage:
num_extremes = check_extreme_cases(df)