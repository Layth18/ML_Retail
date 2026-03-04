import pandas as pd
import joblib
import os
import numpy as np

# ==========================================
# 1. SETUP PATHS & LOAD ASSETS
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "marketing_timeline_model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "marketing_timeline_scaler.pkl")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "MarketingTimelineData", "X_Test_Marketing.csv")
TARGET_PATH = os.path.join(PROJECT_ROOT, "data", "MarketingTimelineData", "y_Test_Marketing.csv")

# Mapping Persona IDs to names
persona_names = {0: "Loyal High-Spender", 1: "Recent Explorer", 2: "At-Risk / Hibernating"}

# ==========================================
# 2. MAIN FUNCTION
# ==========================================
def run_marketing_analysis(add_noise=True, noise_level=0.05):
    # Validate file existence
    for path in [MODEL_PATH, SCALER_PATH, DATA_PATH]:
        if not os.path.exists(path):
            print(f"❌ Missing file: {path}")
            return

    # Load assets
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    X_scaled = pd.read_csv(DATA_PATH)
    y_actual = pd.read_csv(TARGET_PATH)

    # ==========================================
    # 3. PREDICTIONS
    # ==========================================
    predictions = model.predict(X_scaled)

    # Optionally add small Gaussian noise
    if add_noise:
        predictions += np.random.normal(0, noise_level * np.mean(predictions), size=predictions.shape)

    # ==========================================
    # 4. INVERSE SCALE NUMERIC COLUMNS
    # ==========================================
    # Get exactly the columns scaler was trained on
    numeric_cols_for_scaler = list(scaler.feature_names_in_)

    # Ensure all expected columns exist
    missing_cols = set(numeric_cols_for_scaler) - set(X_scaled.columns)
    if missing_cols:
        raise ValueError(f"Missing columns required by scaler: {missing_cols}")

    # Apply inverse transform safely
    df_numeric_unscaled = pd.DataFrame(
        scaler.inverse_transform(X_scaled[numeric_cols_for_scaler]),
        columns=numeric_cols_for_scaler
    )

    # Keep remaining columns (one-hot etc.)
    dummy_cols = [c for c in X_scaled.columns if c not in numeric_cols_for_scaler]

    df_final = pd.concat([df_numeric_unscaled, X_scaled[dummy_cols]], axis=1)

    # Add predictions and actuals
    df_final['Predicted_Gain'] = predictions
    df_final['Actual_Monetary'] = y_actual.values

    # ==========================================
    # 5. DECODE ONE-HOT COLUMNS (Persona, Season, Region)
    # ==========================================
    def decode_label(row, prefix):
        cols = [c for c in df_final.columns if c.startswith(prefix)]
        for c in cols:
            if row[c] > 0.9:  # handle floating point rounding
                return c.replace(prefix, "")
        return None

    df_final['Persona_ID_Raw'] = df_final.apply(lambda r: decode_label(r, 'Pers_'), axis=1)
    df_final['Best_Season'] = df_final.apply(lambda r: decode_label(r, 'Season_'), axis=1)
    df_final['Target_Region'] = df_final.apply(lambda r: decode_label(r, 'Reg_'), axis=1)

    # Safely convert Persona ID
    df_final = df_final.dropna(subset=['Persona_ID_Raw'])
    df_final['Persona_ID'] = df_final['Persona_ID_Raw'].astype(float).astype(int)

    # ==========================================
    # 6. STRATEGY REPORT
    # ==========================================
    print("\n" + "="*80)
    print(f"{'PERSONA GROUP':<25} | {'PEAK SEASON':<12} | {'BEST REGION':<12} | {'EST. GAIN'}")
    print("-" * 80)

    strategy = df_final.groupby('Persona_ID').agg({
        'Best_Season': lambda x: x.value_counts().index[0],
        'Target_Region': lambda x: x.value_counts().index[0],
        'Predicted_Gain': 'mean'
    }).reset_index()

    for _, row in strategy.iterrows():
        name = persona_names.get(row['Persona_ID'], f"ID {row['Persona_ID']}")
        print(f"{name:<25} | {row['Best_Season']:<12} | {row['Target_Region']:<12} | ${row['Predicted_Gain']:>10.2f}")

    print("="*80)

    # ==========================================
    # 7. TEMPORAL INSIGHTS: Weekend vs Weekday
    # ==========================================
    print("\n📅 TEMPORAL INSIGHTS: Weekend Purchase Ratio Impact")
    high_weekend = df_final[df_final['WeekendPurchaseRatio'] > 0.5]['Predicted_Gain'].mean()
    low_weekend = df_final[df_final['WeekendPurchaseRatio'] <= 0.5]['Predicted_Gain'].mean()

    print(f"Average Gain (Weekend Focused): ${high_weekend:.2f}")
    print(f"Average Gain (Weekday Focused): ${low_weekend:.2f}")

    print("\n✅ Marketing Timeline Analysis Complete. Use 'Peak Season' for campaign scheduling.")

# ==========================================
# 8. RUN
# ==========================================
if __name__ == "__main__":
    run_marketing_analysis()