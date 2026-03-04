import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ==============================
# 1. Load Test Data
# ==============================
data_dir = "data/TestTrainData/"

X_test = pd.read_csv(os.path.join(data_dir, "X_Test.csv"))
y_test = pd.read_csv(os.path.join(data_dir, "y_Test.csv")).values.ravel()

print(f"✅ Loaded test data: {X_test.shape}")

# ==============================
# 2. Load Trained Model
# ==============================
model_path = "models/churn_predictor_v1.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError("❌ Trained model not found!")

model = joblib.load(model_path)
print("✅ Model loaded successfully")

# ==============================
# 3. Predictions
# ==============================
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)

# ==============================
# 4. Evaluation Metrics
# ==============================
print("\n" + "="*45)
print("📊 MODEL TEST PERFORMANCE")
print("="*45)

# Multi-class ROC-AUC (One-vs-Rest)
auc = roc_auc_score(
    y_test,
    y_probs,
    multi_class='ovr',
    average='macro'
)

print(f"ROC-AUC (OVR, Macro): {auc:.4f}")

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

# ==============================
# 5. Confusion Matrix
# ==============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False
)

plt.title("Confusion Matrix – Churn Risk Prediction")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# ==============================
# 6. Feature Importance (The Leak Finder)
# ==============================
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    feature_names = X_test.columns
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
    plt.title("Which feature is 'leaking' the answer?")
    plt.xlabel("Importance Score (0 to 1)")
    plt.tight_layout()
    plt.show()

    # Check for suspicious dominance
    if feat_imp.iloc[0] > 0.85:
        print(f"🚨 ALERT: Feature '{feat_imp.index[0]}' has {feat_imp.iloc[0]:.2%} importance.")
        print("This is almost certainly a data leak. Check if this column is a result of churning.")