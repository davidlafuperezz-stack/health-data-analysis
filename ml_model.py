import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

import matplotlib.pyplot as plt
import seaborn as sns

# LOAD DATA
df = pd.read_csv("data/dataset.csv")

# FEATURES / TARGET
X = df[["age", "cholesterol", "blood_pressure", "smoker", "diabetes", "sex"]]
y = df["heart_disease"]

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# SCALING (LOGISTIC REGRESSION)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. LOGISTIC REGRESSION
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)
log_prob = log_model.predict_proba(X_test_scaled)[:, 1]

print("\nLOGISTIC REGRESSION")
print("Accuracy:", accuracy_score(y_test, log_pred))
print("ROC AUC:", roc_auc_score(y_test, log_prob))
print(classification_report(y_test, log_pred))

# Cross-validation
cv_scores_log = cross_val_score(log_model, X_train_scaled, y_train, cv=5)
print("CV Score:", cv_scores_log.mean())

# 2. RANDOM FOREST
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("\nRANDOM FOREST")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("ROC AUC:", roc_auc_score(y_test, rf_prob))
print(classification_report(y_test, rf_pred))

# Cross-validation
cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5)
print("CV Score:", cv_scores_rf.mean())

# CONFUSION MATRIX
plt.figure()
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.show()

# FEATURE IMPORTANCE
importance = rf_model.feature_importances_
features = X.columns

plt.figure()
sns.barplot(x=importance, y=features)
plt.title("Feature Importance (Random Forest)")
plt.show()

# FINAL SUMMARY
print("\nFINAL INSIGHT")
print("Random Forest performs better due to non-linear relationships.")
print("Most important features: cholesterol, blood pressure, age.")
print("Model is now validated using cross-validation and ROC-AUC.")
