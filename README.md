# 📉 Telco Customer Churn Prediction

An end-to-end machine learning portfolio project for identifying telecom customers at higher risk of churn, from exploratory analysis and feature engineering through model tuning, explainability, and a Streamlit prediction prototype.

---

## Business Problem

Customer churn can create significant revenue pressure for subscription businesses. The objective of this project is to identify customers at higher risk of leaving so retention teams could prioritize proactive interventions.

**Key question:** Given a customer's contract, services, tenure, and billing characteristics, can we identify whether that customer is at higher risk of churn?

---

## Dataset

- **Source:** IBM Telco Customer Churn Dataset
- **Size:** 7,043 customer records
- **Columns:** 21 columns including the target variable
- **Target:** `Churn` (Yes / No)
- **Observed churn rate:** approximately **26.5%**

---

## Project Workflow

```
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Encoding & Scaling
      ↓
Stratified Train/Test Split
      ↓
Baseline Model Comparison
      ↓
Class-Imbalance Handling
      ↓
XGBoost Hyperparameter Tuning
      ↓
Model Evaluation
      ↓
SHAP Explainability
      ↓
Streamlit Prototype
```

---

## Key EDA Findings

- Month-to-month customers showed substantially higher churn than customers on longer contracts.
- Fiber-optic customers showed higher churn than DSL customers.
- Customers who churned had shorter average tenure.
- Customers who churned had higher average monthly charges.

These are associations observed in the dataset and should not be interpreted as causal effects.

---

## Feature Engineering

Two additional features were created:

- **`TotalServices`** — number of subscribed telecom/add-on services.
- **`Billing_average`** — total charges divided by tenure, with zero-tenure handling.

Categorical variables were encoded before modeling, and numerical model inputs were standardized using `StandardScaler`.

---

## Baseline Models

The notebook compares four classification approaches on the same stratified train/test split:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Class imbalance is handled using `class_weight='balanced'` for supported scikit-learn models and `scale_pos_weight` for XGBoost.

The notebook reports Accuracy, Precision, Recall, and F1-score for the churn class so model selection is not based on accuracy alone.

---

## XGBoost Hyperparameter Tuning

The XGBoost model is tuned with `GridSearchCV` using:

- 5-fold cross-validation
- `scoring='recall'`
- candidate values for `n_estimators`, `max_depth`, `learning_rate`, and `subsample`

The previously evaluated tuned model achieved approximately:

- **Recall (churn): 0.80**
- **Precision (churn): 0.51**
- **F1-score (churn): 0.62**
- **Cross-validation recall: ~0.82**

On the held-out test set, the confusion matrix was:

- True Negatives: 740
- False Positives: 295
- False Negatives: 73
- True Positives: 301

### Why prioritize recall?

In this business framing, failing to identify a customer who will actually churn (a false negative) may be more costly than contacting some customers who would have stayed. Recall is therefore treated as the primary tuning metric, while Precision and F1-score are still monitored because higher recall increases false positives.

---

## Model Explainability

SHAP is used with `TreeExplainer` to examine:

- global feature influence through a summary plot
- individual predictions through a waterfall plot

This helps connect model behavior back to customer and service characteristics instead of treating the model as a black box.

---

## Streamlit Prototype

The repository includes an interactive Streamlit prototype for customer-level churn prediction.

The app collects the customer fields required to reconstruct the engineered and encoded feature vector used by the trained model, including customer profile, telecom services, contract type, tenure, and billing information.

The application is a **portfolio prototype**, not a production decision system. The displayed model score is not probability-calibrated.

---

## Repository Structure

```
Telco-customer-churn-prediction/
├── Data.csv
├── Telco Customer Churn.ipynb
├── app.py
├── churn_model.pkl
├── scaler.pkl
├── feature_columns.pkl
└── README.md
```

---

## Tech Stack

- **Data Analysis:** pandas, numpy
- **Machine Learning:** scikit-learn, XGBoost
- **Explainability:** SHAP
- **Visualization:** matplotlib, seaborn
- **Prototype Deployment:** Streamlit
