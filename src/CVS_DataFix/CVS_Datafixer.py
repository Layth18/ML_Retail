import pandas as pd
import numpy as np
import os
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings


df= pd.read_csv('data/preparedData/cleaned_data.csv')



# Suppress the deprecation warnings to keep the console clean
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def prepare_data(df, output_path='../data/preparedData/prepared_data.csv'):
    initial_rows = df.shape[0]
    print("\n" + "="*60)
    print(f"🚀 DATA PREPARATION PIPELINE STARTING")
    print(f"   Initial Dataset: {initial_rows} rows | {df.shape[1]} columns")
    print("="*60)

    # --- 1. Missing Values ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    print(f"📊 [STEP 1] Numeric Imputation: ... COMPLETE (Median)")

    # --- 2. Invalid Data Cleaning ---
    total_removed = 0
    total_placeholders = 0
    
    # Selecting columns that are strings/objects
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in categorical_cols:
        invalid_mask = df[col].astype(str).str.lower().isin(['?', 'none', 'null', 'nan', '', 'nan'])
        invalid_count = invalid_mask.sum()
        invalid_pct = invalid_count / len(df)
        
        if 0 < invalid_pct < 0.10:
            df = df[~invalid_mask].copy()
            total_removed += invalid_count
        elif invalid_pct >= 0.10:
            df[col] = df[col].replace(['?', 'None', 'null', 'nan', '', 'NaN'], 'Unknown').fillna('Unknown')
            total_placeholders += 1
            
    print(f"🧹 [STEP 2] Invalid Data Cleaning: .. COMPLETE")
    print(f"   >> Removed {total_removed} rows | Flagged {total_placeholders} cols as 'Unknown'")

    # --- 3. Normalized Dates ---
    if 'RegistrationDate' in df.columns:
        # Added format='mixed' to solve the UserWarning
        df['RegistrationDate'] = pd.to_datetime(df['RegistrationDate'], errors='coerce', format='mixed')
        df['Reg_Year'] = df['RegistrationDate'].dt.year
        df['Reg_Month'] = df['RegistrationDate'].dt.month
        df['Reg_Day'] = df['RegistrationDate'].dt.day
        df = df.drop(columns=['RegistrationDate'])
        print(f"📅 [STEP 3] Date Normalization: ..... COMPLETE (Y/M/D Split)")

    # --- 4. Handle IP (Full Parse) ---
    if 'LastLoginIP' in df.columns:
        # Split the IP into 4 separate parts
        ip_split = df['LastLoginIP'].astype(str).str.split('.', expand=True)
        
        # Rename columns and convert to numeric (filling errors with 0)
        for i in range(4):
            df[f'IP_Octet_{i+1}'] = pd.to_numeric(ip_split[i], errors='coerce').fillna(0).astype(int)
        
        # Drop the original IP column
        df = df.drop(columns=['LastLoginIP'])
        print(f"🌐 [STEP 4] IP Parsing: ............. COMPLETE (4-Octet Split)")

    # --- 5. Categorical Encoding (Integer Mapping) ---
    le = LabelEncoder()
    cat_to_encode = df.select_dtypes(include=['object', 'string']).columns
    for col in cat_to_encode:
        if col != 'CustomerID':
            df[col] = le.fit_transform(df[col].astype(str))
    print(f"🔢 [STEP 5] Label Encoding: ......... COMPLETE (Integer mapping)")

    # --- 6. Cyclical Normalization ---
    cyclical_map = {'PreferredHour': 23, 'PreferredMonth': 12, 'PreferredDayOfWeek': 6}
    for col, max_val in cyclical_map.items():
        if col in df.columns:
            df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
            df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
            df = df.drop(columns=[col])
    print(f"🔄 [STEP 6] Cyclical Mapping: ....... COMPLETE (Sine/Cosine)")

    # --- 7. Skewness (Log Transform) ---
    skew_targets = ['MonetaryTotal', 'Frequency', 'TotalQuantity']
    for col in skew_targets:
        if col in df.columns:
            shift = abs(df[col].min()) + 1 if df[col].min() <= 0 else 0
            df[col] = np.log1p(df[col] + shift)
    print(f"📈 [STEP 7] Skewness Correction: .... COMPLETE (Log1p)")

    # --- SAVE ---
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
    except PermissionError:
        print("\n❌ ERROR: Could not save file! Please close 'prepared_data.csv' if it is open in Excel.")
        return None
    
    print("="*60)
    print(f"✨ FINAL REPORT")
    print(f"   Rows Remaining: {df.shape[0]} (Lost {initial_rows - df.shape[0]} during cleaning)")
    print(f"   Final Columns:  {df.shape[1]}")
    print(f"   File Saved:     {output_path}")
    print("="*60 + "\n")
    
    return df

# Usage
df_prepared = prepare_data(df)
output_path = "data/preparedData/prepared_data.csv"
df_prepared.to_csv(output_path, index=False)

print(f"\n💾 Prepared data saved to: {output_path}")