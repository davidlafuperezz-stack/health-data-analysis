import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv("data/dataset.csv")

# -----------------------
# FEATURES / TARGET
# -----------------------
X = df[["age", "cholesterol", "blood_pressure", "smoker", "diabetes", "sex"]]
y = df["heart_disease"]

# -----------------------
# TRAIN / TEST SPLIT
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------
# FEATURE SCALING (IMPORTANT FOR LOGISTIC REGRESSION)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#LOGISTIC REGRESSION
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

log_pred = log_model.predict(X_test_scaled)

print("\n===== LOGISTIC REGRESSION =====")
print("Accuracy:", accuracy_score(y_test, log_pred))
print(classification_report(y_test, log_pred))

# Confusion Matrix
plt.figure()
sns.heatmap(confusion_matrix(y_test, log_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression - Confusion Matrix")
plt.show()

#RANDOM FOREST 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\n===== RANDOM FOREST =====")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# Confusion Matrix
plt.figure()
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt="d", cmap="Greens")
plt.title("Random Forest - Confusion Matrix")
plt.show()

# FEATURE IMPORTANCE 
importance = rf_model.feature_importances_
features = X.columns
plt.figure()
sns.barplot(x=importance, y=features)
plt.title("Feature Importance (Random Forest)")
plt.show()

# FINAL SUMMARY
print("\nFINAL INSIGHT:")
print("Random Forest captures non-linear relationships better than Logistic Regression.")
print("Most important factors are likely cholesterol, blood pressure, and age.")
