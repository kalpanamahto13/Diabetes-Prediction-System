
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
from io import BytesIO

# Custom CSS for professional styling
def local_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 1rem;
            
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #3b82f6;
    }
    .input-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 0 0 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #6b7280;
    }
    .risk-high {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left-color: #ef4444;
    }
    .risk-low {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left-color: #10b981;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1e40af;
        margin-bottom: 1rem;
    }
    .feature-card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e2e8f0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
       width: 300px;
        font-size: 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e3a8a 100%);
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
                
    }
    .footer {
        width: 100%;
        color: #6b7280;
        font-size: 1.1rem;
        margin-top: 2rem;
        left: 0;
    }
    .built-with {
        text-align: center;
        font-size: 1rem;
        color: #9ca3af;
        margin-top: 1rem;
    }
    .copyright {
        text-align: center;
        font-size: 0.8rem;
        color: #9ca3af;
        margin-top: 0.5rem;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
local_css()

def load_model_and_scaler():
    """
    Load the trained model and scaler.
    """
    try:
        model = joblib.load('models/diabetes_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def get_user_input():
    """
    Get user input for all features.
    """
    st.sidebar.markdown('<p class="sidebar-header">🏥 Medical Information</p>', unsafe_allow_html=True)

    with st.sidebar.expander("📊 Patient Details", expanded=True):
        pregnancies = st.slider('Number of Pregnancies', 0, 17, 1, help="Number of times pregnant")
        age = st.slider('Age (years)', 21, 81, 30, help="Patient's age in years")

    with st.sidebar.expander("🩸 Blood Tests", expanded=True):
        glucose = st.slider('Glucose Level (mg/dL)', 0, 200, 100, help="Plasma glucose concentration")
        insulin = st.slider('Insulin Level (mu U/ml)', 0, 846, 79, help="2-Hour serum insulin")

    with st.sidebar.expander("⚖️ Physical Measurements", expanded=True):
        bmi = st.slider('BMI', 0.0, 67.1, 25.0, 0.1, help="Body mass index")
        skin_thickness = st.slider('Skin Thickness (mm)', 0, 99, 20, help="Triceps skin fold thickness")
        blood_pressure = st.slider('Blood Pressure (mm Hg)', 0, 122, 70, help="Diastolic blood pressure")

    with st.sidebar.expander("🧬 Genetic Factors", expanded=True):
        diabetes_pedigree = st.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.372, 0.001, help="Diabetes pedigree function")

    features = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age
    }

    return features

def make_prediction(model, scaler, features):
    """
    Make prediction using the model.
    """
    try:
        # Convert features to DataFrame
        input_df = pd.DataFrame([features])

        # Scale the input
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        return prediction, probability
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

def display_prediction(prediction, probability):
    """
    Display the prediction result.
    """
    if prediction == 1:
        st.error(f"High Risk of Diabetes (Probability: {probability*100:.1f}%)")
    else:
        st.success(f"Low Risk of Diabetes (Probability: {probability*100:.1f}%)")

def risk_factor_analysis(features, probability):
    """
    Provide risk factor analysis.
    """
    st.subheader("Risk Factor Analysis")

    high_risk_factors = []
    medium_risk_factors = []

    # Glucose
    if features['Glucose'] > 140:
        high_risk_factors.append("High Glucose Level (>140 mg/dL)")
    elif features['Glucose'] > 100:
        medium_risk_factors.append("Elevated Glucose Level (>100 mg/dL)")

    # BMI
    if features['BMI'] > 30:
        high_risk_factors.append("Obesity (BMI > 30)")
    elif features['BMI'] > 25:
        medium_risk_factors.append("Overweight (BMI > 25)")

    # Age
    if features['Age'] > 45:
        medium_risk_factors.append("Age > 45 years")

    # Blood Pressure
    if features['BloodPressure'] > 90:
        medium_risk_factors.append("High Blood Pressure (>90 mm Hg)")

    # Pregnancies
    if features['Pregnancies'] > 5:
        medium_risk_factors.append("Multiple Pregnancies (>5)")

    # Insulin
    if features['Insulin'] > 200:
        medium_risk_factors.append("High Insulin Level (>200 mu U/ml)")

    if high_risk_factors:
        st.error("🚨 High Risk Factors:")
        for factor in high_risk_factors:
            st.write(f"• {factor}")

    if medium_risk_factors:
        st.warning("⚠️ Medium Risk Factors:")
        for factor in medium_risk_factors:
            st.write(f"• {factor}")

    if not high_risk_factors and not medium_risk_factors:
        st.success("✅ No significant risk factors identified")

def generate_report(features, prediction, probability):
    """
    Generate a downloadable prediction report.
    """
    report = f"""
    DIABETES PREDICTION REPORT
    ==========================

    Patient Information:
    -------------------
    Pregnancies: {features['Pregnancies']}
    Glucose: {features['Glucose']} mg/dL
    Blood Pressure: {features['BloodPressure']} mm Hg
    Skin Thickness: {features['SkinThickness']} mm
    Insulin: {features['Insulin']} mu U/ml
    BMI: {features['BMI']}
    Diabetes Pedigree Function: {features['DiabetesPedigreeFunction']}
    Age: {features['Age']}

    Prediction Results:
    ------------------
    Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}
    Probability: {probability*100:.1f}%

    Risk Assessment:
    ---------------
    Risk Level: {'High' if probability > 0.7 else 'Medium' if probability > 0.4 else 'Low'}

    Recommendations:
    ---------------
    {'Please consult a healthcare professional for further evaluation.' if prediction == 1 else 'Maintain healthy lifestyle and regular check-ups.'}

    Note: This is an AI-based prediction and should not replace professional medical advice.
    """

    return report

def download_report(report):
    """
    Create a download button for the report.
    """
    b64 = base64.b64encode(report.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="diabetes_prediction_report.txt">📥 Download Prediction Report</a>'
    st.markdown(href, unsafe_allow_html=True)

def main():
    """
    Main Streamlit application.
    """
    # Header with logo and title
    st.markdown('<h1 class="main-header">🏥 Diabetes Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Diabetes Risk Assessment</p>', unsafe_allow_html=True)

    # Load model and scaler
    model, scaler = load_model_and_scaler()

    if model is None or scaler is None:
        st.error("Failed to load model. Please ensure the model is trained first.")
        st.info("Run `python src/train_model.py` to train the model.")
        return

    # Get user input
    features = get_user_input()

    # Display input features in a card
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.subheader("📋 Input Features")
    features_df = pd.DataFrame([features])
    st.dataframe(features_df, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

    # Prediction section below input
    st.subheader("🔍 Risk Assessment")

    # Make prediction
    if st.button("🔍 Predict Diabetes Risk", type="primary"):
        with st.spinner("🔬 Analyzing your health data..."):
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
            prediction, probability = make_prediction(model, scaler, features)

        if prediction is not None:
            # Enhanced prediction display
            risk_class = "risk-high" if prediction == 1 else "risk-low"
            st.markdown(f'<div class="prediction-card {risk_class}">', unsafe_allow_html=True)
            if prediction == 1:
                st.markdown("### 🚨 High Risk of Diabetes")
                st.markdown(f"**Probability:** {probability*100:.1f}%")
                st.markdown("**Recommendation:** Please consult a healthcare professional for further evaluation.")
            else:
                st.markdown("### ✅ Low Risk of Diabetes")
                st.markdown(f"**Probability:** {probability*100:.1f}%")
                st.markdown("**Recommendation:** Maintain healthy lifestyle and regular check-ups.")
            st.markdown('</div>', unsafe_allow_html=True)

            # Risk factor analysis in expander
            with st.expander("📊 Detailed Risk Factor Analysis"):
                risk_factor_analysis(features, probability)

            # Generate and download report in expander
            with st.expander("📄 Prediction Report"):
                report = generate_report(features, prediction, probability)
                st.text_area("Report Preview", report, height=300)
                download_report(report)

    # Footer
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown("----------------------")
    st.markdown("**Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice.  Always consult a healthcare provider for medical concerns. ")
    st.markdown('</div>', unsafe_allow_html=True)

    # built-with 
    st.markdown("<div class='built-with'>"
                "Built with ❤️ using Streamlit and Machine Learning"
                "</div>", unsafe_allow_html=True)
    st.markdown("<div class='copyright'>" 
    "© copyright 2025 - Diabeties Prediction System by Kalpana Mahto . All rights reserved."
               "</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
