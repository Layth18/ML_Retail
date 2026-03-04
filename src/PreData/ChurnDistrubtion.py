import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. Load Data (Keeping your exact path) ---
data_path = 'data/TestTrainData/y_Test.csv'

if not os.path.exists(data_path):
    print(f"❌ File not found at {data_path}")
else:
    # Load the data
    y_data = pd.read_csv(data_path)
    
    # --- 2. Target the Churn Column ---
    # Based on your previous output, column 0 is CustomerID. 
    # We take column 1 which should be the Churn label.
    target_col = y_data.columns[0] 
    
    print(f"✅ Analyzing column: '{target_col}'")

    # --- 3. Calculate Frequencies ---
    counts = y_data[target_col].value_counts().sort_index()
    percentages = y_data[target_col].value_counts(normalize=True).sort_index() * 100

    print("\n" + "="*35)
    print(f"📊 CHURN FREQUENCY (0-3)")
    print("="*35)
    summary = pd.DataFrame({'Count': counts, 'Percentage (%)': percentages.round(2)})
    print(summary)
    print("="*35)

    # --- 4. Visualize ---
    plt.figure(figsize=(8, 5))
    # Fixed the warning by assigning hue and x
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="magma", legend=False)
    
    # Add counts on top of bars
    for i, v in enumerate(counts.values):
        ax.text(i, v + (max(counts.values)*0.02), str(v), ha='center', fontweight='bold')

    plt.title(f"Distribution of {target_col} in Raw Data")
    plt.xlabel("Churn Class (0-3)")
    plt.ylabel("Number of Customers")
    
    # Adding descriptive labels if they match your business logic
    plt.xticks(ticks=[0, 1, 2, 3], labels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
    
    plt.tight_layout()
    plt.show()