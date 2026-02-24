import pandas as pd


df=pd.read_csv('data/preparedData/cleaned_data.csv')

def validate_dataset_logic(df):
    report = []
    total = len(df)
    
    # helper function to log errors
    def log_err(col, count, issue):
        if count > 0:
            report.append({
                'Feature': col,
                'Invalid Count': count,
                'Invalid %': f"{(count/total)*100:.2f}%",
                'Issue': issue
            })

    # --- 1. Out of Bounds (Intervalle) Checks ---
    # Numeric constraints based on your table
    intervals = {
        'Recency': (0, 400),
        'Frequency': (1, 50),
        'MonetaryTotal': (-5000, 15000),
        'MonetaryAvg': (5, 500),
        'CustomerTenure': (0, 730),
        'PreferredHour': (0, 23),
        'WeekendRatio': (0.0, 1.0),
        'ReturnRatio': (0.0, 1.0),
        'Age': (18, 81),
        'Satisfaction': (0, 5) # Note: table says -1, 0, 1-5, 99. Usually 99 is a placeholder.
    }

    for col, (low, high) in intervals.items():
        if col in df.columns:
            invalid = df[(df[col] < low) | (df[col] > high)].shape[0]
            log_err(col, invalid, f"Outside [{low}, {high}]")

    # --- 2. Specific Placeholder Checks ---
    # Many of your features use 99, 999, or -1 as "Inconnu"
    placeholders = {
        'SupportTickets': [-1, 999],
        'Satisfaction': [-1, 99],
        'AgeCategory': ['Inconnu'],
        'LoyaltyLevel': ['Inconnu'],
        'BasketSize': ['Inconnu'],
        'Gender': ['Unknown']
    }

    for col, vals in placeholders.items():
        if col in df.columns:
            count = df[df[col].isin(vals)].shape[0]
            log_err(col, count, f"Uses placeholders {vals}")

    # --- 3. Consistency Logic ---
    # e.g., FirstPurchase should logically be >= CustomerTenure
    if 'FirstPurchase' in df.columns and 'CustomerTenure' in df.columns:
        logic_violation = (df['FirstPurchase'] < df['CustomerTenure']).sum()
        log_err('FirstPurchase', logic_violation, "Less than Tenure (Logic Error)")

    # --- 4. Constant Value Check (Step 50: Newsletter) ---
    if 'Newsletter' in df.columns:
        if df['Newsletter'].nunique() <= 1:
            log_err('Newsletter', total, "Constant value (Should be suppressed)")

    # Display Report
    validation_df = pd.DataFrame(report)
    print("--- 🛠️ DATA FIXER: INVALID DATA REPORT ---")
    if not validation_df.empty:
        print(validation_df.to_string(index=False))
    else:
        print("✅ All data strictly follows the provided interval rules!")
    
    return validation_df

# Execute
validation_report = validate_dataset_logic(df)