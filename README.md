# 🏦 Loan Approval Prediction using Machine Learning

A complete end-to-end **Machine Learning** project for predicting loan approval status based on customer and loan information. The project covers the entire ML workflow, including data preprocessing, feature engineering, exploratory data analysis (EDA), model comparison, hyperparameter tuning, and model deployment.

---

## 📌 Project Overview

Loan approval is one of the most critical decisions made by financial institutions. Incorrect decisions may increase financial risk, while overly conservative policies may reject qualified applicants.

This project develops a predictive machine learning model to classify whether a loan application will be **Approved** or **Rejected** using applicant demographics, financial history, and loan characteristics.

---

## 🎯 Objectives

- Perform comprehensive Exploratory Data Analysis (EDA)
- Handle missing values and outliers
- Engineer informative features
- Build robust preprocessing pipelines
- Compare multiple machine learning algorithms
- Optimize models using hyperparameter tuning
- Evaluate models using multiple performance metrics
- Save a production-ready pipeline for deployment

---

# 📂 Dataset

This project uses the **Loan Default Dataset** from Kaggle.

**Dataset Link**

👉 https://www.kaggle.com/datasets/yasserh/loan-default-dataset

### Dataset Information

| Property | Value |
|----------|-------|
| Source | Kaggle |
| Dataset | Loan Default Dataset |
| Rows | ~148,000 |
| Features | 34 |
| Problem Type | Binary Classification |
| Target | Status |

### Target Variable

| Value | Meaning |
|-------|---------|
| 0 | Rejected |
| 1 | Approved |

The dataset contains borrower demographics, loan characteristics, financial information, property details, credit history, and application-related attributes. It also contains missing values and class imbalance, making it suitable for demonstrating a complete machine learning workflow.

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- LightGBM
- Imbalanced-Learn
- Joblib

---

# 📊 Exploratory Data Analysis (EDA)

The project includes extensive EDA, including:

- Target distribution
- Missing value analysis
- Outlier detection
- Numerical feature distributions
- Categorical feature analysis
- Correlation heatmap
- Feature relationships
- Default rate analysis
- Class imbalance visualization

---

# ⚙️ Data Preprocessing

The preprocessing pipeline includes:

- Missing value imputation
  - Median (Numerical)
  - Most Frequent (Categorical)
- Standardization using StandardScaler
- Label Encoding for binary categorical variables
- One-Hot Encoding for nominal categorical variables
- Feature Engineering
- Random Under Sampling for handling class imbalance

---

# 🧠 Feature Engineering

Several domain-specific features were created to improve predictive performance:

- Benchmark Interest Rate
- Interest Verification
- Loan-to-Income Ratio
- Loan-to-Property Ratio
- Monthly Payment Estimation
- Payment-to-Income Ratio

These engineered features helped improve model performance by incorporating meaningful financial relationships.

---

# 🤖 Machine Learning Models

The following algorithms were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- HistGradientBoosting
- AdaBoost
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- XGBoost
- LightGBM

---

# 🔍 Hyperparameter Tuning

Hyperparameter optimization was performed using:

- RandomizedSearchCV
- GridSearchCV

Models tuned include:

- Decision Tree
- Random Forest
- XGBoost

---

# 📈 Model Evaluation

Models were evaluated using multiple metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve
- Stratified K-Fold Cross Validation

---

# 🔄 Cross Validation

To ensure reliable performance estimation and reduce overfitting, **Stratified 5-Fold Cross Validation** was applied.

Evaluation metrics include:

- Mean Accuracy
- Mean Precision
- Mean Recall
- Mean F1-Score
- Mean ROC-AUC

---

# 📁 Project Structure

```
Loan-Approval-Prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Model_Training.ipynb
│   ├── 05_Model_Comparison.ipynb
│   └── 06_Hyperparameter_Tuning.ipynb
│
├── models/
│   └── loan_pipeline.pkl
│
├── images/
│   ├── target_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Loan-Approval-Prediction.git
```

Navigate to the project directory

```bash
cd Loan-Approval-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Run the notebooks in the following order:

1. Exploratory Data Analysis
2. Data Cleaning
3. Feature Engineering
4. Data Preprocessing
5. Model Training
6. Hyperparameter Tuning
7. Model Evaluation
8. Model Saving

---

# 📊 Visualizations

The project includes several visualizations:

- Missing Values
- Target Distribution
- Numerical Feature Distributions
- Categorical Feature Analysis
- Correlation Heatmap
- Feature Importance
- Confusion Matrix
- ROC Curve
- Model Comparison

---

# 💾 Saving the Model

The final production pipeline can be saved using:

```python
import joblib

joblib.dump(model, "loan_pipeline.pkl")
```

Load the model later using:

```python
model = joblib.load("loan_pipeline.pkl")
```

---

# 📌 Results

The final optimized model achieved strong predictive performance after applying:

- Feature Engineering
- Hyperparameter Optimization
- Class Imbalance Handling
- Stratified Cross Validation

Performance was evaluated using multiple complementary metrics to ensure robust generalization and reduce overfitting.

---

# 🔮 Future Improvements

- Explainability using SHAP
- Feature Selection
- Stacking Ensemble Models
- Optuna Hyperparameter Optimization
- Streamlit Web Application
- Docker Deployment
- CI/CD Pipeline

---

# 👨‍💻 Author

**Ahmed Bebars**

Computer and Control Systems Engineering Student

### Areas of Interest

- Artificial Intelligence
- Machine Learning
- Data Science
- Computer Vision

**GitHub:** https://github.com/ahmedbebars1

**Kaggle:** https://www.kaggle.com/ahmedbebars1

**LinkedIn:** *Add your LinkedIn profile here*

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub. Your support helps others discover the project and motivates future improvements.
