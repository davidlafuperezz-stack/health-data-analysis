# Health Data Analysis & Machine Learning Project

## Overview
This project is a complete data science pipeline applied to a synthetic healthcare dataset. It includes exploratory data analysis (EDA), data visualization, and machine learning models to predict the presence of heart disease based on patient health indicators.

The goal is to simulate a real-world data science workflow from raw data to predictive modeling.

---

## Dataset
The dataset includes the following features:

- Age
- Sex
- Cholesterol level
- Blood pressure
- Smoking status
- Diabetes status
- Heart disease (target variable)

---

## Project Structure

### 1. Exploratory Data Analysis (EDA)
Performed in `analysis.py`:

- Data inspection and structure analysis
- Missing value check
- Statistical summaries
- Correlation analysis
- Group-based risk factor analysis
- Data visualization:
  - Histograms
  - Boxplots
  - Barplots
  - Pairplot
  - Correlation heatmap

Key insights:
- Higher cholesterol is associated with heart disease
- Blood pressure increases risk
- Smoking and diabetes significantly impact probability
- Age shows a clear correlation with risk

---

### 2. Machine Learning Models
Implemented in `ml_model.py`:

- Logistic Regression
- Random Forest Classifier

Evaluation metrics:
- Accuracy score
- Classification report (precision, recall, F1-score)
- Confusion matrix
- Feature importance analysis

---

## Key Results
- Random Forest performs better in capturing non-linear relationships than Logistic Regression
- Most important predictive features:
  - Cholesterol
  - Blood pressure
  - Age

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## What I learned
- Data cleaning and exploratory analysis
- Feature relationships and correlation analysis
- Machine learning model training and evaluation
- Model comparison and interpretation
- Data visualization for storytelling

---

## Future Improvements
- Hyperparameter tuning with GridSearchCV
- Cross-validation
- ROC-AUC evaluation
- Deployment using Streamlit or FastAPI

---

## Author
David (Data Science / Machine Learning learner)

GitHub: https://github.com/davidlafuperezz-stack
