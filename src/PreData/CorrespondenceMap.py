import pandas as pd
import json
import os

# Load both datasets
raw_df = pd.read_csv('data/preparedData/cleaned_data.csv')
encoded_df = pd.read_csv('data/preparedData/prepared_data.csv')

# Features we want to map
categorical_features = ['FavoriteSeason', 'CustomerType', 'RFMSegment', 'Region', 'ChurnRiskCategory']

mapping_report = {}

for col in categorical_features:
    if col in raw_df.columns and col in encoded_df.columns:
        # Create a unique mapping of Label -> Number
        # We dropna to ensure we don't map nulls
        pairs = pd.DataFrame({
            'label': raw_df[col].astype(str),
            'value': encoded_df[col]
        }).drop_duplicates().sort_values('value')
        
        # Convert to dictionary { "0": "Summer", "1": "Winter" ... }
        mapping_report[col] = dict(zip(pairs['value'].astype(int).astype(str), pairs['label']))

# Save as JSON for the Flask Dashboard
os.makedirs('data/metadata', exist_ok=True)
with open('data/metadata/category_mapping.json', 'w') as f:
    json.dump(mapping_report, f, indent=4)

print("✅ Correspondence Map created in data/metadata/category_mapping.json")
# Print a preview
print(json.dumps(mapping_report, indent=2))