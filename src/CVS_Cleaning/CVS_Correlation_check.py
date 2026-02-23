import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('data/raw_data.csv')

def analyze_and_plot_correlation(df, threshold=0.8):
    # 1. Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # 2. Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # 3. Save the full matrix to CSV
    corr_matrix.to_csv('correlation_matrix.csv')
    print("Full correlation matrix saved to 'correlation_matrix.csv'.")
    
    # 4. Create the Heatmap Plot
    plt.figure(figsize=(12, 10))
    # 'mask' hides the upper triangle to make the plot easier to read (since it's a mirror image)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, 
                mask=mask, 
                annot=True,          # Show the numbers in the squares
                fmt=".2f",           # Round to 2 decimals
                cmap='coolwarm',     # Red for positive, Blue for negative
                center=0, 
                linewidths=.5, 
                cbar_kws={"shrink": .8})
    
    plt.title('Feature Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')  # Saves the plot as an image
    print("Heatmap saved as 'correlation_heatmap.png'.\n")
    
    # 5. Identify High Correlation Pairs
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    high_corr_pairs = []
    print(f"--- High Correlation Pairs (Threshold > {threshold}) ---")
    for column in upper.columns:
        for row in upper.index:
            val = upper.loc[row, column]
            if abs(val) > threshold:
                print(f"{row} <-> {column}: {val:.4f}")
                high_corr_pairs.append((row, column, val))
                
    if not high_corr_pairs:
        print("No pairs found with high correlation.")
        
    return high_corr_pairs

# Usage:
high_corr_list = analyze_and_plot_correlation(df, threshold=0.8)