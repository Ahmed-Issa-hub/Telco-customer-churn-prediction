import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Telco Churn Prediction", page_icon="📉", layout="centered")

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("📉 Telco Customer Churn Prediction")
st.caption("Portfolio prototype — predictions depend on the trained model and are not a production decision system.")

st.subheader("Customer Profile")

gender = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])

phone_service = st.selectbox("Phone Service", ["No", "Yes"])
if phone_service == "No":
    multiple_lines = "No phone service"
else:
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])

internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

if internet_service == "No":
    online_security = "No internet service"
    online_backup = "No internet service"
    device_protection = "No internet service"
    tech_support = "No internet service"
    streaming_tv = "No internet service"
    streaming_movies = "No internet service"
else:
    online_security = st.selectbox("Online Security", ["No", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])

contract_type = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
)

tenure_months = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0, step=0.05)
total_charges = st.number_input("Total Charges", min_value=0.0, value=float(monthly_charges * max(tenure_months, 1)), step=1.0)

def set_if_present(frame, column, value):
    if column in frame.columns:
        frame.loc[0, column] = value

def build_model_input():
    x = pd.DataFrame([np.zeros(len(feature_columns))], columns=feature_columns)

    # Numeric/original features
    set_if_present(x, "SeniorCitizen", 1 if senior_citizen == "Yes" else 0)
    set_if_present(x, "tenure", tenure_months)
    set_if_present(x, "MonthlyCharges", monthly_charges)
    set_if_present(x, "TotalCharges", total_charges)

    # Feature engineering used in training
    billing_average = total_charges / (tenure_months if tenure_months > 0 else 1)

    service_values = [
        phone_service,
        multiple_lines,
        internet_service,
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies,
    ]
    no_values = {"No", "No internet service", "No phone service"}
    total_services = sum(value not in no_values for value in service_values)

    set_if_present(x, "Billing_average", billing_average)
    set_if_present(x, "TotalServices", total_services)

    # Contract encoding used during training
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    set_if_present(x, "Contract", contract_map[contract_type])

    # One-hot encoded categorical features (drop_first=True during training)
    set_if_present(x, "gender_Male", int(gender == "Male"))
    set_if_present(x, "Partner_Yes", int(partner == "Yes"))
    set_if_present(x, "Dependents_Yes", int(dependents == "Yes"))
    set_if_present(x, "PhoneService_Yes", int(phone_service == "Yes"))

    set_if_present(x, "MultipleLines_No phone service", int(multiple_lines == "No phone service"))
    set_if_present(x, "MultipleLines_Yes", int(multiple_lines == "Yes"))

    set_if_present(x, "InternetService_Fiber optic", int(internet_service == "Fiber optic"))
    set_if_present(x, "InternetService_No", int(internet_service == "No"))

    for prefix, value in [
        ("OnlineSecurity", online_security),
        ("OnlineBackup", online_backup),
        ("DeviceProtection", device_protection),
        ("TechSupport", tech_support),
        ("StreamingTV", streaming_tv),
        ("StreamingMovies", streaming_movies),
    ]:
        set_if_present(x, f"{prefix}_No internet service", int(value == "No internet service"))
        set_if_present(x, f"{prefix}_Yes", int(value == "Yes"))

    set_if_present(x, "PaymentMethod_Credit card (automatic)", int(payment_method == "Credit card (automatic)"))
    set_if_present(x, "PaymentMethod_Electronic check", int(payment_method == "Electronic check"))
    set_if_present(x, "PaymentMethod_Mailed check", int(payment_method == "Mailed check"))
    set_if_present(x, "PaperlessBilling_Yes", int(paperless_billing == "Yes"))

    return x

if st.button("Predict churn risk", type="primary"):
    model_input = build_model_input()
    model_input_scaled = scaler.transform(model_input)
    prediction = model.predict(model_input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ The model classifies this customer as likely to churn.")
    else:
        st.success("✅ The model classifies this customer as likely to stay.")

    if hasattr(model, "predict_proba"):
        churn_score = model.predict_proba(model_input_scaled)[0][1]
        st.metric("Model churn score", f"{churn_score:.1%}")
        st.caption("This score is model output and has not been probability-calibrated.")
