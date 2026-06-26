import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# LOAD DATA
df = pd.read_csv("data/dataset.csv")

# FEATURES (X) AND TARGET (y)
X = df[["age", "cholesterol", "blood_pressure", "smoker", "diabetes", "sex"]]
y = df["heart_disease"]

# TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# MODEL
model = LogisticRegression()

model.fit(X_train, y_train)

# PREDICTIONS
y_pred = model.predict(X_test)

# EVALUATION
accuracy = accuracy_score(y_test, y_pred)

print("MODEL ACCURACY:", accuracy)
print("\nCLASSIFICATION REPORT:\n")
print(classification_report(y_test, y_pred))
