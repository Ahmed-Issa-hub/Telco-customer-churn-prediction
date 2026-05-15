import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. تحميل النموذج
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. عنوان التطبيق
st.title("توقع مغادرة العملاء")

# 3. المدخلات (نفس اللي عملته)
contract_type = st.selectbox("اختر نوع عقد العميل:", options=["Month-to-month", "One year"])
internetService = st.selectbox("اختر نوع خدمة الإنترنت:", options=["Fiber optic", "DSL", "No"])
payment_method = st.selectbox("اختر طريقة الدفع:", options=["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])
tenure_months = st.slider("حدد عدد أشهر اشتراك العميل (Tenure):", min_value=0, max_value=72, value=12)
MonthlyCharges = st.number_input("أدخل قيمة الفاتورة الشهرية:", min_value=0.0, max_value=200.0, step=0.05)

user_data = {
    'Contract': contract_type,
    'tenure': tenure_months,
    'MonthlyCharges': MonthlyCharges,
    'InternetService': internetService,
    'PaymentMethod': payment_method
    }

# 4. زر التوقع
if st.button("توقع"):
    columns = joblib.load('feature_columns.pkl')
    
    # ابدأ بـ DataFrame فاضي بالأعمدة الصح
    xgb_input = pd.DataFrame([np.zeros(len(columns))], columns=columns)
    
    # امليه مباشرة بالقيم المحولة
    xgb_input['tenure'] = tenure_months
    xgb_input['MonthlyCharges'] = MonthlyCharges
    
    # Contract: Label Encoding
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    xgb_input['Contract'] = contract_map[contract_type]
    
    # InternetService: One-Hot — ابحث عن اسم العمود الصح في feature_columns
    # InternetService
    if internetService == 'Fiber optic':
        xgb_input['InternetService_Fiber optic'] = 1
    elif internetService == 'No':
        xgb_input['InternetService_No'] = 1
    
    # PaymentMethod
    if payment_method == 'Electronic check':
        xgb_input['PaymentMethod_Electronic check'] = 1
    elif payment_method == 'Mailed check':
        xgb_input['PaymentMethod_Mailed check'] = 1
    elif payment_method == 'Credit card (automatic)':
        xgb_input['PaymentMethod_Credit card (automatic)'] = 1
    
    xgb_input_scaled = scaler.transform(xgb_input)
    prediction = model.predict(xgb_input_scaled)
    
    if prediction[0] == 1:
        st.error("⚠️ العميل على الأرجح سيغادر")
    else:
        st.success("✅ العميل على الأرجح سيبقى")
 
