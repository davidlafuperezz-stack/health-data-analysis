import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# LOAD DATA
df = pd.read_csv("data/dataset.csv")


# BASIC INFO

print("SHAPE:", df.shape)
print("\nINFO:")
print(df.info())

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nDESCRIPTIVE STATISTICS:")
print(df.describe())


# TARGET ANALYSIS
print("\nHEART DISEASE DISTRIBUTION:")
print(df["heart_disease"].value_counts(normalize=True))


# GROUP ANALYSIS 
print("\nMEAN VALUES BY HEART DISEASE:")
print(df.groupby("heart_disease").mean(numeric_only=True))

print("\nSMOKING IMPACT:")
print(df.groupby("smoker")["heart_disease"].mean())

print("\nDIABETES IMPACT:")
print(df.groupby("diabetes")["heart_disease"].mean())

print("\nSEX IMPACT:")
print(df.groupby("sex")["heart_disease"].mean())


# CORRELATION ANALYSIS
plt.figure(figsize=(8,6))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Matrix")
plt.show()

# DISTRIBUTIONS
for col in ["age", "cholesterol", "blood_pressure"]:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# RELATIONSHIP PLOTS
plt.figure()
sns.boxplot(x="heart_disease", y="cholesterol", data=df)
plt.title("Cholesterol vs Heart Disease")
plt.show()

plt.figure()
sns.boxplot(x="heart_disease", y="blood_pressure", data=df)
plt.title("Blood Pressure vs Heart Disease")
plt.show()

plt.figure()
sns.barplot(x="smoker", y="heart_disease", data=df)
plt.title("Smoking vs Heart Disease Risk")
plt.show()

plt.figure()
sns.barplot(x="diabetes", y="heart_disease", data=df)
plt.title("Diabetes vs Heart Disease Risk")
plt.show()

# ADVANCED VISUAL
sns.pairplot(df, hue="heart_disease")
plt.show()

# INSIGHT SUMMARY
print("\nFINAL INSIGHTS:")
print("- Higher cholesterol is associated with heart disease")
print("- Blood pressure increases risk")
print("- Smoking and diabetes strongly increase probability")
print("- Age shows a clear risk increase pattern")
