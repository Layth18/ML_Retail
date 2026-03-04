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
# Constants / Columns for new marketing model
# ─────────────────────────────────────────────
numeric_cols = ["Recency", "Frequency", "CustomerTenureDays", "WeekendPurchaseRatio"]
season_cols = ["Season_0","Season_1","Season_2","Season_3"]
region_cols = ["Reg_4","Reg_8","Reg_Other"]
persona_cols = ["Pers_0","Pers_1","Pers_2","Pers_3"]

season_names = {0: "Autumn", 1: "Winter", 2: "Spring", 3: "Summer"}
region_names = {"Reg_4": "Central Europe", "Reg_8": "UK", "Reg_Other": "Other"}

persona_names_map = {
    0: "Loyal High-Spender",
    2: "At-Risk / Hibernating",
    1: "Recent Explorer / Newbie",
    3: "Loyal Active"

}

churn_names_map = {0: "Critique", 1: "Faible", 2: "Moyen", 3: "Élevé"}
season_bonus_multiplier = 1.2  # 20% gain boost for clients favoring the season

marketing_input_cols = numeric_cols + season_cols + region_cols + persona_cols

# ─────────────────────────────────────────────
# /api/predict_all
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# /api/predict_all
# ─────────────────────────────────────────────
@app.route('/api/predict_all', methods=['POST'])
def predict_all():
    try:
        data = request.get_json()
        features = data["features"]

        # Only 3 features
        feature_names = ["Recency", "Frequency", "CustomerTenureDays"]
        input_df = pd.DataFrame([features], columns=feature_names)

        # Scale features
        features_scaled = scaler.transform(input_df)

        persona_idx = int(persona_model.predict(features_scaled)[0])
        churn_idx = int(churn_model.predict(features_scaled)[0])

        return jsonify({
            "persona_name": persona_names_map[persona_idx],
            "churn_risk": churn_names_map[churn_idx],
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
        csv_path = os.path.join(BASE_DIR, "..", "data", "MarketingTimelineData", "X_Train_Marketing.csv")
        customers = pd.read_csv(csv_path)

        # ── Columns used by the marketing model
        numeric_cols = ["Recency","Frequency","CustomerTenureDays","WeekendPurchaseRatio"]
        season_cols  = ["Season_0","Season_1","Season_2","Season_3"]
        region_cols  = ["Reg_4","Reg_8","Reg_Other"]
        persona_cols = ["Pers_0","Pers_1","Pers_2","Pers_3"]
        all_cols = numeric_cols + season_cols + region_cols + persona_cols

        # Build model input
        X = pd.DataFrame(0, index=customers.index, columns=all_cols)
        for col in all_cols:
            if col in customers.columns:
                X[col] = customers[col].fillna(0)

        # Scale numeric features only
        X_scaled = X.copy()
        X_scaled[numeric_cols] = marketing_scaler.transform(X[numeric_cols])

        # Predict base gain
        customers["predicted_gain"] = marketing_model.predict(X_scaled)

        # ── Season Aggregation ──
        season_names = {0: "Autumn", 1: "Winter", 2: "Spring", 3: "Summer"}
        season_bonus_multiplier = 1.2
        seasons_agg = {}

        for s_idx, s_name in season_names.items():
            gain_col = f"season_{s_idx}_gain"
            season_col = f"Season_{s_idx}"

            # Copy base gain and apply bonus if client prefers this season
            customers[gain_col] = customers["predicted_gain"]
            customers.loc[customers[season_col] > 0.8, gain_col] *= season_bonus_multiplier

            # Aggregate total for season
            season_total = customers[gain_col].sum()

            # Top regions
            top_regions = []
            for region_col, region_label in {"Reg_4": "Central Europe", "Reg_8": "UK", "Reg_Other": "Other"}.items():
                region_gain = customers.loc[customers[region_col] > 0.8, gain_col].sum()
                top_regions.append({"region": region_label, "estimated_gain": round(region_gain,2)})

            # Sort top 3 regions
            top_regions.sort(key=lambda x: x["estimated_gain"], reverse=True)
            seasons_agg[s_name] = {"season_total": round(season_total,2), "top_regions": top_regions[:3]}

        # ── Region Aggregation ──
        regions_view = []
        for region_col, region_label in {"Reg_4": "Central Europe", "Reg_8": "UK", "Reg_Other": "Other"}.items():
            s_gains = []
            season_gain_cols = [f"season_{s_idx}_gain" for s_idx in season_names]
            for s_idx, s_name in season_names.items():
                gain = customers.loc[customers[region_col] > 0.8, f"season_{s_idx}_gain"].sum()
                s_gains.append({"season": s_name, "estimated_gain": round(gain,2)})
            total_annual_gain = customers.loc[customers[region_col] > 0.8, season_gain_cols].sum().sum()
            regions_view.append({
                "region": region_label,
                "seasons_ranked": sorted(s_gains, key=lambda x: x["estimated_gain"], reverse=True),
                "total_annual_gain": round(total_annual_gain,2)
            })
        regions_view.sort(key=lambda x: x["total_annual_gain"], reverse=True)

        return jsonify({"seasons": seasons_agg, "regions": regions_view})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)