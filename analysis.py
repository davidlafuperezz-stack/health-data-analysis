import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/dataset.csv")

# BASIC INFO
print("DATA PREVIEW")
print(df.head())

print("\nDATA INFO")
print(df.info())

print("\nDESCRIPTIVE STATISTICS")
print(df.describe())


# BASIC ANALYSIS
print("\nHeart disease rate:")
print(df["heart_disease"].value_counts(normalize=True))

print("\nSmokers vs heart disease:")
print(df.groupby("smoker")["heart_disease"].mean())

print("\nDiabetes vs heart disease:")
print(df.groupby("diabetes")["heart_disease"].mean())

print("\nRisk factors summary:")
print(df.groupby("heart_disease")[["cholesterol", "blood_pressure", "age"]].mean())


# CORRELATION HEATMAP
plt.figure(figsize=(6,4))
sns.heatmap(
    df[["age","cholesterol","blood_pressure","smoker","diabetes","heart_disease"]].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Key Feature Correlations")
plt.show()


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

# Blood pressure vs heart disease
plt.figure()
sns.boxplot(x="heart_disease", y="blood_pressure", data=df)
plt.title("Blood Pressure vs Heart Disease")
plt.show()

# Smoking effect
plt.figure()
sns.barplot(x="smoker", y="heart_disease", data=df)
plt.title("Smoking vs Heart Disease Probability")
plt.show()

# Cholesterol distribution by class
plt.figure()
sns.histplot(data=df, x="cholesterol", hue="heart_disease", kde=True)
plt.title("Cholesterol Distribution by Heart Disease")
plt.show()

# Pairplot (very powerful visual)
sns.pairplot(df, hue="heart_disease")
plt.show()

# INSIGHT
print("\nINSIGHT:")
print("Heart disease is more frequent in patients with higher cholesterol, higher blood pressure, smoking habits, and diabetes.")
print("These factors show clear patterns in the dataset and may indicate strong risk relationships.")
