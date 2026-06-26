import pandas as pd

# Load dataset
df = pd.read_csv("data/dataset.csv")
# Show first rows
print("DATA PREVIEW")
print(df.head())
# Basic info
print("\nDATA INFO")
print(df.info())
# Basic statistics
print("\nDESCRIPTIVE STATISTICS")
print(df.describe())
# Simple insights
print("\nAVERAGE AGE")
print(df["age"].mean())
print("\nHEART DISEASE COUNT")
print(df["heart_disease"].value_counts())
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of age
plt.figure()
sns.histplot(df["age"], bins=10)
plt.title("Age Distribution")
plt.show()

# Cholesterol vs heart disease
plt.figure()
sns.boxplot(x="heart_disease", y="cholesterol", data=df)
plt.title("Cholesterol vs Heart Disease")
plt.show()
print("\nINSIGHT:")
print("Patients with higher cholesterol tend to show more heart disease cases (based on this small dataset).")
