# 📉 Telco Customer Churn Prediction

A machine learning project that predicts whether a telecom customer will churn, built end-to-end from data exploration to a deployed Streamlit application.

---

## 🔍 Problem Statement

Customer churn is one of the most critical challenges in the telecom industry. Losing a customer costs significantly more than retaining one. This project builds a model that identifies customers likely to churn **before** they leave, enabling proactive retention strategies.

**Key Question:** Given a customer's contract details, services, and billing information — will they churn?

---

## 📊 Dataset

- **Source:** IBM Telco Customer Churn Dataset
- **Size:** 7,043 customers × 21 features
- **Target:** `Churn` (Yes / No) — 23.4% churn rate (imbalanced)

---

## 🧪 Project Pipeline

```
Data Loading & Cleaning
        ↓
Exploratory Data Analysis (EDA)
        ↓
Feature Engineering
        ↓
Model Training & Comparison
        ↓
Hyperparameter Tuning (GridSearchCV)
        ↓
Model Explainability (SHAP)
        ↓
Streamlit Deployment
```

---

## 📈 Exploratory Data Analysis — Key Findings

| Feature | Insight |
|---|---|
| **Contract Type** | Month-to-month customers churn at **37%** vs 6% for two-year contracts |
| **Internet Service** | Fiber optic customers churn **6% more** than DSL customers |
| **Tenure** | Customers who churned had avg tenure of **18 months** vs 37 months for retained |
| **Monthly Charges** | Churned customers paid avg **$74/month** vs $61 for retained |

---

## ⚙️ Feature Engineering

Two new features were created based on EDA insights:

- **`TotalServices`** — Total number of add-on services subscribed (e.g. StreamingTV, OnlineBackup...)
- **`Billing_average`** — Average monthly spend relative to tenure (`TotalCharges / tenure`)

Encoding strategy:
- **Label Encoding** → `Contract` (ordinal: Month-to-month < One year < Two year)
- **One-Hot Encoding** → `InternetService`, `PaymentMethod`, and all other categorical features

---

## 🤖 Model Comparison

Four models were trained with `class_weight='balanced'` / `scale_pos_weight` to handle class imbalance:

| Model | Recall (Churn) | F1 (Churn) |
|---|---|---|
| Logistic Regression | 0.78 | 0.61 |
| Decision Tree | 0.79 | 0.63 |
| Random Forest | 0.71 | 0.62 |
| **XGBoost (selected)** | **0.69 → 0.80*** | **0.62** |

*After Hyperparameter Tuning via GridSearchCV

> **Why Recall?** In churn prediction, missing a customer who will leave (False Negative) is more costly than a false alarm. Recall was chosen as the primary metric.

---

## 🔧 Hyperparameter Tuning

Used `GridSearchCV` with `cv=5` and `scoring='recall'`:


**Result:** Recall improved from **0.69 → 0.80** (+11%)

---

## 🧠 Model Explainability — SHAP Values

SHAP (SHapley Additive exPlanations) was used to explain model decisions globally and per customer.

**Top 3 most influential features:**
1. `Contract` — Month-to-month contracts strongly push toward churn
2. `InternetService_Fiber optic` — Fiber optic users are higher risk
3. `tenure` — Shorter tenure increases churn probability

---

## 🚀 Streamlit App

An interactive web app that takes customer details as input and predicts churn in real time.

**Input features:**
- Contract type
- Internet service type
- Payment method
- Tenure (months)
- Monthly charges

**Output:**
- ✅ Customer is likely to stay
- ⚠️ Customer is likely to churn

---

## 🗂️ Repository Structure

```
📁 telco-customer-churn-prediction
├── Telco_Customer_Churn.ipynb   # Full analysis notebook
├── app.py                        # Streamlit application
├── churn_model.pkl               # Trained XGBoost model
├── scaler.pkl                    # Fitted StandardScaler
├── feature_columns.pkl           # Feature column names
└── README.md
```

---

## 🛠️ Tech Stack


- **Data:** pandas, numpy
- **Modeling:** scikit-learn, XGBoost
- **Explainability:** SHAP
- **Deployment:** Streamlit
- **Visualization:** matplotlib, seaborn

---

## 👤 Author


[LinkedIn](https://www.linkedin.com/in/ahmed-eissa-837691a1/)
