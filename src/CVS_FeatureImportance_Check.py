import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Setup Data
df = pd.read_csv('data/preparedData/prepared_data.csv')
X = df.drop(columns=['CustomerID', 'ChurnRiskCategory'], errors='ignore')
y = df['ChurnRiskCategory']

# Split for Permutation Importance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize for PCA/RFE
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

def normalize(s):
    return (s - s.min()) / (s.max() - s.min())

print("🧪 Running Ultimate Feature Selection (5 Algorithms)...")

# --- 1. Random Forest Importance ---
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
score_rf = normalize(pd.Series(rf.feature_importances_, index=X.columns))

# --- 2. RFE (Linear) ---
rfe = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=1).fit(X_scaled, y)
score_rfe = normalize(1 / pd.Series(rfe.ranking_, index=X.columns))

# --- 3. PCA (Variance) ---
pca = PCA(n_components=10).fit(X_scaled)
pca_loadings = np.mean(np.abs(pca.components_), axis=0)
score_pca = normalize(pd.Series(pca_loadings, index=X.columns))

# --- 4. Mutual Information (Statistical) ---
mi = mutual_info_classif(X, y, random_state=42)
score_mi = normalize(pd.Series(mi, index=X.columns))

# --- 5. Permutation Importance (Impact) ---
perm = permutation_importance(rf, X_test, y_test, n_repeats=5, random_state=42)
score_perm = normalize(pd.Series(perm.importances_mean, index=X.columns))

# --- COMBINE & RANK ---
ultimate_score = (score_rf + score_rfe + score_pca + score_mi + score_perm) / 5
ultimate_score = ultimate_score.sort_values(ascending=False) * 100

# --- FINAL REPORT ---
print("\n" + "💎" * 20)
print("  THE ULTIMATE ELITE 10 FEATURES")
print("💎" * 20)
for i, (feature, score) in enumerate(ultimate_score.head(10).items(), 1):
    print(f"{i:2d}. {feature:<25} | Score: {score:.2f}/100")
print("💎" * 20)

# Visualize
plt.figure(figsize=(12, 7))
ultimate_score.head(15).plot(kind='barh', color='crimson').invert_yaxis()
plt.title("Ultimate Elite Feature Ranking (Consensus of 5 Algos)")
plt.xlabel("Ensemble Power Score (0-100)")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()