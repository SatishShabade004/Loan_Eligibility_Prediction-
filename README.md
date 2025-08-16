# Loan Approval Prediction

A clean, end-to-end machine learning workflow for predicting **loan approval** using Python. This project
covers data loading, exploratory data analysis (EDA), preprocessing, modeling, and evaluation — all contained
in the Jupyter notebook: **`loan-prediction (1).ipynb`**.

## 🎯 Problem Statement
Given applicant details (income, credit history, dependents, property area, etc.), predict whether a loan
application will be **approved** or **rejected** (binary classification).
```

(If you have CSVs, place them alongside the notebook, e.g. `loan-test.csv, loan-train.csv`.)

## 🧠 Approach

1. **EDA**  
   - Inspect missing values, class balance, and distributions.
   - Visualize key relationships (e.g., CreditHistory vs Loan_Status, Income vs Approval).

2. **Data Cleaning & Preprocessing**  
   - Impute missing values appropriately (median/most frequent).
   - Encode categoricals (One-Hot Encoding).
   - Scale numerical features where helpful (e.g., StandardScaler).
   - Train/Validation split to avoid leakage.

3. **Modeling**  
   - Start with a strong baseline (Logistic Regression).
   - Try tree-based models for non-linear relations (Decision Tree / Random Forest / Gradient Boosting).
   - Compare models using accuracy/F1 and cross-validation.
   - Selected models observed in notebook: **DecisionTreeClassifier, GradientBoostingClassifier, LogisticRegression, RandomForestClassifier**.

4. **Evaluation**  
   - Primary metrics: **Accuracy** (plus F1 for imbalance), Confusion Matrix, ROC-AUC.
   - Example metric logs detected from the notebook:
```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
model=LogisticRegression()
model.fit(X_train, Y_train)
y_train_pred=model.predict(X_train)
train_accuracy=accuracy_score(Y_trai
confusion_matrix(Y_test,y_test_pred)
confusion_matrix(Y_train,y_train_pred)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train, Y_train)
y_train_pred=model.predict(X_train)
train_accuracy=accuracy_score(Y_train, y_train_pred)
prin
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test,y_test_pred)
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_train,y_train_pred)
https://scikit-learn.org/stable/modules/preprocessing.html
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
y_test_pred=model.predict(X_test)
test_accuracy=accuracy_score(Y_test, y_test_pred)
print(test_accuracy)
y_test_pred=model.predict(X_test)
test_accuracy=accurac
```

## 📊 Features & Target
- **Target**: `Loan_Status` (Y/N).  
- **Typical Features** (examples): `Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome,
CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area`.

> If your exact column names differ, adjust in the preprocessing cell in the notebook.

## 🛠️ Environment & Requirements

Create a fresh environment and install dependencies:

```bash
# using pip
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U numpy pandas scikit-learn matplotlib seaborn
# optionally, if used:
pip install xgboost catboost lightgbm imbalanced-learn
```

## ▶️ How to Run

1. Place the dataset CSV files next to the notebook (e.g., `loan-test.csv, loan-train.csv`).  
2. Open the notebook:
   - VS Code / Jupyter: open **`loan-prediction (1).ipynb`** and run cells top-to-bottom.
3. At the end, review evaluation metrics and confusion matrix to assess performance.

## ✅ Results (Example)
- Baseline Logistic Regression achieves a reasonable starting **accuracy**; tree models may improve recall on minority class.
- Use cross-validation to confirm robustness and avoid overfitting.

_Your exact numbers depend on dataset splits; see the **Evaluation** section of the notebook for printed scores._

## 🚀 Next Steps
- Hyperparameter tuning (GridSearchCV/RandomizedSearchCV).
- Address class imbalance (class weights / SMOTE).
- Feature importance (model-specific) and SHAP for explainability.
- Export the best model with `joblib` for deployment.
- Optional: Build a simple Flask/Streamlit app for predictions.

## 📎 Notes
- If loading paths fail, update the `pd.read_csv(...)` paths in the first data-loading cell.
- Keep random seeds fixed for reproducibility (e.g., `random_state=42`).

---

