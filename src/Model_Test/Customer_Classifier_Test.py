import pandas as pd
import joblib
import os
import numpy as np

# ==========================================
# 1. SETUP PATHS & LOAD ASSETS
# ==========================================
# Navigate from src/Model_Test/ to Project Root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "persona_classifier.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "main_scaler.pkl")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "TestTrainData", "X_Train.csv")

# ==========================================
# 2. DEFINE BUSINESS MAPPING
# ==========================================
persona_map = {
    0: {"name": "Loyal High-Spender"},
    1: {"name": "Recent Explorer / Newbie"},
    2: {"name": "At-Risk / Hibernating"},
    3: {"name": "Loyal Active"},
}

def run_analysis():
    # Validation
    for path in [MODEL_PATH, SCALER_PATH, DATA_PATH]:
        if not os.path.exists(path):
            print(f"❌ Critical Error: Missing file at {path}")
            return

    # Load everything
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    X_scaled = pd.read_csv(DATA_PATH)

       # ==========================================
    # 3. PREDICT & UNSCALE
    # ==========================================
    # Predict clusters based on scaled data
    clusters = model.predict(X_scaled)

    # Step A: Reverse the Scaling (Z-score to Logged values)
    unscaled_array = scaler.inverse_transform(X_scaled)
    df_final = pd.DataFrame(unscaled_array, columns=X_scaled.columns)
    
    # Step B: Reverse the Log Transformation (Log1p to Real Units)
    # UPDATED: Changed 'frequency' to 'Frequency' to match your column name
    skewed_cols = ['MonetaryTotal', 'Frequency'] 
    for col in skewed_cols:
        if col in df_final.columns:
            df_final[col] = np.expm1(df_final[col])

    df_final['ClusterID'] = clusters


    # ==========================================
    # 4. GENERATE SUMMARY REPORT
    # ==========================================
    print("\n" + "="*80)
    print(f"{'PERSONA NAME':<30} | {'COUNT':<10} | {'POPULATION %':<12}")
    print("-" * 80)

    counts = df_final['ClusterID'].value_counts().sort_index()
    total_users = len(df_final)

    for cluster_id, count in counts.items():
        info = persona_map.get(cluster_id, {"name": f"Unknown {cluster_id}", "strategy": "N/A"})
        percent = (count / total_users) * 100
        
        print(f"{info['name']:<30} | {count:<10} | {percent:>10.1f}%")

    print("="*80)

    # ==========================================
    # 5. ACTUAL BUSINESS MEANS (REAL UNITS)
    # ==========================================
    print("\n📊 BUSINESS INSIGHTS: Average Metrics per Persona (Real Units)")
    
    # Calculate means and round for clean display
    report = df_final.groupby('ClusterID').mean().round(2)
    
    # Replace numeric index with Persona Names for the final table
    report.index = [persona_map.get(i, {"name": i})["name"] for i in report.index]
    
    print(report)
    print("\n✅ Analysis Complete. Use these means to verify your persona mappings match the data behaviors.")

if __name__ == "__main__":
    run_analysis()