import pandas as pd


import pandas as pd

df = pd.read_csv('data/preparedData/cleaned_data.csv')

def normalize_dates(df, date_col='RegistrationDate'):
    # Try parsing with dd-mm-yy format only
    parsed = pd.to_datetime(
        df[date_col],
        format='%d-%m-%y',
        errors='coerce'
    )

    total = len(df)
    valid = parsed.notna().sum()
    invalid = parsed.isna().sum()

    print(f"📅 Date format check: dd-mm-yy")
    print(f"Total rows: {total}")
    print(f"✔ Valid dates: {valid}")
    print(f"❌ Invalid / need change: {invalid}")
    print(f"Mismatch rate: {(invalid/total)*100:.2f}%")

    return parsed

# Execute
df_normalized = normalize_dates(df)