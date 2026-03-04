import pandas as pd
from xgboost import XGBClassifier
import joblib
import os

X_train = pd.read_csv("data/TestTrainData/X_Train.csv")
y_train = pd.read_csv("data/TestTrainData/y_Train.csv").values.ravel()

model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)

# This will now definitely have 7 features
joblib.dump(model, 'models/churn_predictor_v1.pkl')
print(f"✅ Model trained on {X_train.shape[1]} features.")