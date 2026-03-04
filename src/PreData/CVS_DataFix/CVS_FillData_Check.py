import pandas as pd
import numpy as np

df = pd.read_csv('data/preparedData/cleaned_data.csv')


def analyze_missing_stats(df):
    # 1. Identify columns with missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    
    if not missing_cols:
        print("No missing data found! Your dataset is already full.")
        return None

    stats_list = []
    
    for col in missing_cols:
        # Calculate stats
        col_mean = df[col].mean() if np.issubdtype(df[col].dtype, np.number) else "N/A"
        col_median = df[col].median() if np.issubdtype(df[col].dtype, np.number) else "N/A"
        
        # Mode can return multiple values, so we take the first one
        mode_series = df[col].mode()
        col_mode = mode_series[0] if not mode_series.empty else "N/A"
        
        stats_list.append({
            'Column': col,
            'Missing Count': df[col].isnull().sum(),
            'Mean': col_mean,
            'Median': col_median,
            'Mode': col_mode,
            'Type': df[col].dtype
        })
    
    # Create comparison table
    stats_df = pd.DataFrame(stats_list)
    
    print("--- Missing Data Imputation Strategy Table ---")
    print(stats_df.to_string(index=False))
    
    return stats_df

# Execute
missing_stats = analyze_missing_stats(df)