import pandas as pd

# Load your data
df = pd.read_csv('data/raw_data.csv')

def check_missing_data(df):
    # 1. Global Missing Data Percentage
    total_cells = df.size
    total_missing = df.isnull().sum().sum()
    global_percent = (total_missing / total_cells) * 100
    
    print(f"--- Global Statistics ---")
    print(f"Total Missing Values: {total_missing}")
    print(f"Global Missing Percentage: {global_percent:.2f}%\n")
    
    # 2. Per Category (Column) Missing Data Percentage
    print(f"--- Per Category Statistics ---")
    missing_count = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    
    # Combine into a clean table
    missing_df = pd.DataFrame({
        'Missing Count': missing_count,
        'Percentage (%)': missing_percent
    }).sort_values(by='Percentage (%)', ascending=False)
    
    # Filter to show only columns that actually have missing data
    return missing_df[missing_df['Missing Count'] > 0]

# Usage:
missing_report = check_missing_data(df)
print(missing_report)