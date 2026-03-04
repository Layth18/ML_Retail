import pandas as pd
import joblib
import os
from sklearn.cluster import KMeans

# 1. Load Training Data (Scaled)
data_path = 'data/TestTrainData/X_Train.csv'
df_scaled = pd.read_csv(data_path)

# 2. Final KMeans Model
optimal_k = 4

model_kmeans = KMeans(
    n_clusters=optimal_k,
    init='k-means++',
    random_state=42,
    n_init=10
)

# Fit on the scaled training features
clusters = model_kmeans.fit_predict(df_scaled)

# 3. Save Model
os.makedirs('models', exist_ok=True)
joblib.dump(model_kmeans, 'models/persona_classifier.pkl')
print("✅ Persona model saved successfully")

# 4. Summary for Business
summary = df_scaled.copy()
summary['Persona'] = clusters
print("\n🚀 Persona Feature Means (Scaled Units):")
print(summary.groupby('Persona').mean())
