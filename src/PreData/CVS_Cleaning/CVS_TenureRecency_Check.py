import pandas as pd

# Load your data
df = pd.read_csv('data/raw_data.csv')

def check_invalid_customers(df):
    """
    Checks for invalid customer records:
    - Negative Recency
    - Negative Tenure
    - Recency greater than Tenure
    Returns a summary table with counts and percentages.
    """
    total_rows = len(df)
    
    # 1️⃣ Negative Recency
    neg_recency = df[df['Recency'] < 0].shape[0]
    
    # 2️⃣ Negative Tenure
    neg_tenure = df[df['CustomerTenureDays'] < 0].shape[0]
    
    # 3️⃣ Recency greater than Tenure
    rec_gt_tenure = df[df['Recency'] > df['CustomerTenureDays']].shape[0]
    
    # Combine into a summary DataFrame
    summary_df = pd.DataFrame({
        'Issue': ['Negative Recency', 'Negative Tenure', 'Recency > Tenure'],
        'Count': [neg_recency, neg_tenure, rec_gt_tenure],
        'Percentage (%)': [
            (neg_recency / total_rows) * 100,
            (neg_tenure / total_rows) * 100,
            (rec_gt_tenure / total_rows) * 100
        ]
    })
    
    return summary_df

# Usage
invalid_report = check_invalid_customers(df)
print("--- Invalid Customer Data Report ---")
print(invalid_report)