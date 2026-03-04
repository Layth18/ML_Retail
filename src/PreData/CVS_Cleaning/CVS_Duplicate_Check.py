import pandas as pd

df = pd.read_csv('data/raw_data.csv')

def check_duplicate_rows(df):
    # Count the number of duplicated rows
    num_duplicates = df.duplicated().sum()
    
    print(f"--- Duplicate Rows Check ---")
    print(f"Number of duplicate rows found: {num_duplicates}")
    
    # Optional: see the percentage of the dataset that is duplicated
    duplicate_pct = (num_duplicates / len(df)) * 100
    print(f"Percentage of duplicated data: {duplicate_pct:.2f}%")
    
    return num_duplicates

# Usage:
# num_dupes = check_duplicate_rows(df)

# Usage:
num_dupes = check_duplicate_rows(df)