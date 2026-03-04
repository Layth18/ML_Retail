import pandas as pd


def check_overrepresentation(df, threshold=0.95):
    overrepresented_cols = []
    
    for col in df.columns:
        # Get the percentage of the most frequent value
        top_value_pct = df[col].value_counts(normalize=True).iloc[0]
        
        if top_value_pct >= threshold:
            most_freq_val = df[col].mode()[0]
            print(f"Column '{col}' is {top_value_pct*100:.2f}% '{most_freq_val}'")
            overrepresented_cols.append(col)
            
    if not overrepresented_cols:
        print("No columns exceed the threshold.")
    return overrepresented_cols

# Run it
df = pd.read_csv('data/raw_data.csv')
cols_to_drop = check_overrepresentation(df, threshold=0.95)