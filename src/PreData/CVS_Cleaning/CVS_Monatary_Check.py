import pandas as pd

# Load your data
df = pd.read_csv("data/raw_data.csv")

def check_negative_monetary(df):
    if "MonetaryTotal" not in df.columns:
        print("⚠️ Column 'MonetaryTotal' not found in the dataset.")
        return None
    
    # Count rows with negative MonetaryTotal
    neg_rows = df[df["MonetaryTotal"] < 0]
    count_neg = len(neg_rows)
    total_rows = len(df)
    percent_neg = (count_neg / total_rows) * 100 if total_rows > 0 else 0
    
    print(f"--- Negative MonetaryTotal Check ---")
    print(f"Rows with negative MonetaryTotal: {count_neg}")
    print(f"Percentage of dataset: {percent_neg:.2f}%")
    
    # Optionally return the problematic rows
    return neg_rows

# Usage
neg_monetary_rows = check_negative_monetary(df)
if not neg_monetary_rows.empty:
    print(neg_monetary_rows.head())