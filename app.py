import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the XGBoost model
@st.cache_resource
def load_xgboost_model():
    try:
        model = xgb.XGBClassifier()
        model.load_model("churn_model.json")
        return model
    except Exception as e:
        st.error(f"Error loading XGBoost model: {e}")
        st.stop()

# Create encoders (since we don't have the saved ones)
@st.cache_data
def create_encoders():
    encoders = {}
    
    # Define the encoding mappings based on your original data
    encoders['gender'] = {'Female': 0, 'Male': 1}
    encoders['Partner'] = {'No': 0, 'Yes': 1}
    encoders['Dependents'] = {'No': 0, 'Yes': 1}
    encoders['PhoneService'] = {'No': 0, 'Yes': 1}
    encoders['MultipleLines'] = {'No': 0, 'No phone service': 1, 'Yes': 2}
    encoders['InternetService'] = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
    encoders['OnlineSecurity'] = {'No': 0, 'No internet service': 1, 'Yes': 2}
    encoders['OnlineBackup'] = {'No': 0, 'No internet service': 1, 'Yes': 2}
    encoders['DeviceProtection'] = {'No': 0, 'No internet service': 1, 'Yes': 2}
    encoders['TechSupport'] = {'No': 0, 'No internet service': 1, 'Yes': 2}
    encoders['StreamingTV'] = {'No': 0, 'No internet service': 1, 'Yes': 2}
    encoders['StreamingMovies'] = {'No': 0, 'No internet service': 1, 'Yes': 2}
    encoders['Contract'] = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    encoders['PaperlessBilling'] = {'No': 0, 'Yes': 1}
    encoders['PaymentMethod'] = {
        'Bank transfer (automatic)': 0, 
        'Credit card (automatic)': 1, 
        'Electronic check': 2, 
        'Mailed check': 3
    }
    
    return encoders

# Load model and encoders
model = load_xgboost_model()
encoders = create_encoders()

st.title("🔄 Customer Churn Prediction App")
st.markdown("Enter customer details below and click **Predict Churn** to know if the customer is likely to leave.")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    
    st.subheader("Service Information")
    phone = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "No internet service", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"])

with col2:
    st.subheader("Account Information")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
    total_charges = st.slider("Total Charges ($)", 0.0, 9000.0, 2000.0)
    
    st.subheader("Additional Services")
    device_protection = st.selectbox("Device Protection", ["No", "No internet service", "Yes"])
    tech_support = st.selectbox("Tech Support", ["No", "No internet service", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "No internet service", "Yes"])
    
    st.subheader("Contract Details")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", [
        "Bank transfer (automatic)", 
        "Credit card (automatic)", 
        "Electronic check", 
        "Mailed check"
    ])

# Prediction button
if st.button("🔮 Predict Churn", type="primary"):
    try:
        # Create input data
        input_data = {
            'gender': encoders['gender'][gender],
            'SeniorCitizen': senior,
            'Partner': encoders['Partner'][partner],
            'Dependents': encoders['Dependents'][dependents],
            'tenure': tenure,
            'PhoneService': encoders['PhoneService'][phone],
            'MultipleLines': encoders['MultipleLines'][multiple_lines],
            'InternetService': encoders['InternetService'][internet],
            'OnlineSecurity': encoders['OnlineSecurity'][online_security],
            'OnlineBackup': encoders['OnlineBackup'][online_backup],
            'DeviceProtection': encoders['DeviceProtection'][device_protection],
            'TechSupport': encoders['TechSupport'][tech_support],
            'StreamingTV': encoders['StreamingTV'][streaming_tv],
            'StreamingMovies': encoders['StreamingMovies'][streaming_movies],
            'Contract': encoders['Contract'][contract],
            'PaperlessBilling': encoders['PaperlessBilling'][paperless],
            'PaymentMethod': encoders['PaymentMethod'][payment_method],
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Convert to DataFrame with the correct column order
        feature_order = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges'
        ]
        
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_order]  # Ensure correct column order
        
        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        # Display results
        st.markdown("---")
        if prediction[0] == 1:
            st.error("⚠️ **HIGH RISK**: This customer is likely to churn!")
            churn_prob = prediction_proba[0][1] * 100
            st.write(f"**Churn Probability**: {churn_prob:.1f}%")
        else:
            st.success("✅ **LOW RISK**: This customer is likely to stay!")
            stay_prob = prediction_proba[0][0] * 100
            st.write(f"**Retention Probability**: {stay_prob:.1f}%")
        
        # Show probability breakdown
        with st.expander("View Probability Details"):
            st.write(f"**Probability of Staying**: {prediction_proba[0][0]:.3f}")
            st.write(f"**Probability of Churning**: {prediction_proba[0][1]:.3f}")
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.write("Please check your input values and try again.")

# Add some helpful information
with st.expander("ℹ️ How to use this app"):
    st.write("""
    1. Fill in all the customer information in the form above
    2. Click the 'Predict Churn' button
    3. The model will predict whether the customer is likely to churn or stay
    4. Higher churn probability means the customer needs attention to prevent them from leaving
    """)

with st.expander("📊 Model Information"):
    st.write("""
    - **Model Type**: XGBoost Classifier
    - **Features Used**: 19 customer attributes
    - **Training Data**: Telecommunications customer data with SMOTE balancing
    """)