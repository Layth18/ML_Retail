import os
import joblib
import pandas as pd
import numpy as np
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

# ─────────────────────────────────────────────
# App setup & Pathing Logic
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5500", "http://localhost:5500"])

# ─────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────
try:
    MODEL_PATH = os.path.join(BASE_DIR, "..", "models")
    
    churn_model = joblib.load(os.path.join(MODEL_PATH, "churn_predictor_v1.pkl"))
    persona_model = joblib.load(os.path.join(MODEL_PATH, "persona_classifier.pkl"))
    scaler = joblib.load(os.path.join(MODEL_PATH, "main_scaler.pkl"))
    marketing_model = joblib.load(os.path.join(MODEL_PATH, "marketing_timeline_model.pkl"))
    marketing_scaler = joblib.load(os.path.join(MODEL_PATH, "marketing_timeline_scaler.pkl"))
    print("✅ All Models loaded successfully.")
except Exception as e:
    print(f"❌ Critical Error loading files: {e}")
    raise e

# ─────────────────────────────────────────────
# Constants / Columns
# ─────────────────────────────────────────────
numeric_cols = ["Recency", "Frequency", "CustomerTenureDays", "WeekendPurchaseRatio"]
season_cols = ["Season_0.0","Season_1.0","Season_2.0","Season_3.0"]
region_cols = ["Reg_6.0","Reg_12.0","Reg_Other"]
persona_cols = ["Pers_0", "Pers_1", "Pers_2"]

season_names = {0: "Autumn", 1: "Winter", 2: "Spring", 3: "Summer"}
region_names = {"Reg_6.0": "Central Europe", "Reg_12.0": "UK", "Reg_Other": "Other"}
persona_names_map = {0: "Loyal High-Spender", 1: "Recent Explorer", 2: "At-Risk / Hibernating"}
churn_names_map = {0: "Critique", 1: "Faible", 2: "Moyen", 3: "Élevé"}
season_bonus_multiplier = 1.2  # 20% gain boost for targeting favorite season

marketing_input_cols = numeric_cols + season_cols + region_cols + persona_cols

# ─────────────────────────────────────────────
# /api/predict_all
# ─────────────────────────────────────────────
@app.route('/api/predict_all', methods=['POST'])
def predict_all():
    try:
        data = request.get_json()
        features = data["features"]

        feature_names = [
            'Recency', 'Frequency', 'FavoriteSeason',
            'CustomerTenureDays', 'WeekendPurchaseRatio',
            'Region', 'MonetaryTotal'
        ]
        input_df = pd.DataFrame([features], columns=feature_names)
        raw_region = input_df['Region'].iloc[0]

        # Log-transform skewed features
        for col in ['MonetaryTotal', 'Frequency']:
            input_df[col] = np.log1p(input_df[col])

        features_scaled = scaler.transform(input_df)

        persona_idx = int(persona_model.predict(features_scaled)[0])
        churn_idx = int(churn_model.predict(features_scaled)[0])

        # ── Marketing Simulation ──
        best_season = None
        max_gain = float("-inf")

        for s_idx in range(4):
            row = pd.DataFrame(0, index=[0], columns=marketing_input_cols)

            # numeric features
            for col in numeric_cols:
                row[col] = input_df[col].values[0]

            # persona
            row[f'Pers_{persona_idx}'] = 1

            # season
            season_col = f"Season_{float(s_idx)}"
            row[season_col] = 1

            # region
            raw_region_col = f"Reg_{float(raw_region)}"
            if raw_region_col in row.columns:
                row[raw_region_col] = 1

            # scale numeric columns
            row_scaled = row.copy()
            row_scaled[numeric_cols] = marketing_scaler.transform(row[numeric_cols])

            pred = float(marketing_model.predict(row_scaled)[0])

            if pred > max_gain:
                max_gain = pred
                best_season = season_names[s_idx]

        return jsonify({
            "persona_name": persona_names_map[persona_idx],
            "churn_risk": churn_names_map[churn_idx],
            "best_season": best_season,
            "estimated_gain": round(np.expm1(max_gain), 2),  # unskew log prediction
            "raw_indices": {"persona": persona_idx, "churn": churn_idx}
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ─────────────────────────────────────────────
# /api/marketing_dashboard
# ─────────────────────────────────────────────
@app.route("/api/marketing_dashboard", methods=["GET"])
def marketing_dashboard():
    try:
        # Load dataset
        csv_path = os.path.join(BASE_DIR, "..", "data", "MarketingTimelineData", "X_Train_Marketing.csv")
        customers = pd.read_csv(csv_path)

        # Build model input
        all_cols = numeric_cols + season_cols + region_cols + persona_cols
        X = pd.DataFrame(0, index=customers.index, columns=all_cols)
        for col in all_cols:
            if col in customers.columns:
                X[col] = customers[col].fillna(0)

        # Scale numeric features
        X_scaled = X.copy()
        X_scaled[numeric_cols] = marketing_scaler.transform(X[numeric_cols])

        # Predict base gain
        customers["predicted_gain"] = marketing_model.predict(X_scaled)

        # ─── Season section ───
        seasons_agg = {}
        for s_idx, s_name in season_names.items():
            season_col = f"Season_{float(s_idx)}"
            gain_col = f"season_{s_idx}_gain"

            # Apply bonus only for clients favoring this season
            customers[gain_col] = customers["predicted_gain"].copy()
            customers.loc[customers[season_col] > 0.8, gain_col] *= season_bonus_multiplier

            # Aggregate season total
            season_total = customers[gain_col].sum()

            # Top regions in this season
            region_sums = []
            for region_col, region_label in region_names.items():
                gain = customers.loc[customers[region_col] > 0.8, gain_col].sum()
                region_sums.append({"region": region_label, "estimated_gain": round(gain, 2)})
            region_sums.sort(key=lambda x: x["estimated_gain"], reverse=True)

            seasons_agg[s_name] = {"top_regions": region_sums[:3], "season_total": round(season_total, 2)}

        # ─── Region section ───
        regions_view = []
        for region_col, region_label in region_names.items():
            s_gains = []
            season_gain_cols = [f"season_{s_idx}_gain" for s_idx in season_names]
            for s_idx, s_name in season_names.items():
                gain = customers.loc[customers[region_col] > 0.8, f"season_{s_idx}_gain"].sum()
                s_gains.append({"season": s_name, "estimated_gain": round(gain, 2)})

            # Annual estimate for this region
            total_annual_gain = customers.loc[customers[region_col] > 0.8, season_gain_cols].sum().sum()

            regions_view.append({
                "region": region_label,
                "seasons_ranked": sorted(s_gains, key=lambda x: x["estimated_gain"], reverse=True),
                "total_annual_gain": round(total_annual_gain, 2)
            })

        # Sort regions by total annual gain
        regions_view.sort(key=lambda x: x["total_annual_gain"], reverse=True)

        return jsonify({"seasons": seasons_agg, "regions": regions_view})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
    
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)