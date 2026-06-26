import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/dataset.csv")


# BASIC EXPLORATION
print("DATA PREVIEW")
print(df.head())

print("\nDATA INFO")
print(df.info())

print("\nDESCRIPTIVE STATISTICS")
print(df.describe())

# BASIC ANALYSIS
print("\nAVERAGE AGE")
print(df["age"].mean())

print("\nHEART DISEASE COUNT")
print(df["heart_disease"].value_counts())

print("\nAVERAGE CHOLESTEROL BY HEART DISEASE")
print(df.groupby("heart_disease")["cholesterol"].mean())


# VISUALIZATIONS

# Age distribution
plt.figure()
sns.histplot(df["age"], bins=10)
plt.title("Age Distribution")
plt.show()

# Cholesterol vs heart disease
plt.figure()
sns.boxplot(x="heart_disease", y="cholesterol", data=df)
plt.title("Cholesterol vs Heart Disease")
plt.show()

# Cholesterol distribution by class
plt.figure()
sns.histplot(data=df, x="cholesterol", hue="heart_disease", kde=True)
plt.title("Cholesterol Distribution by Heart Disease")
plt.show()

# Correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# INSIGHT

print("\nINSIGHT:")
print("Patients with heart disease tend to have higher cholesterol levels on average.")
print("There is a visible difference in cholesterol distribution between both groups.")
