import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    st.sidebar.header("Medical Information")

    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 1)
    glucose = st.sidebar.slider('Glucose (mg/dL)', 0, 200, 100)
    blood_pressure = st.sidebar.slider('Blood Pressure (mm Hg)', 0, 122, 70)
    skin_thickness = st.sidebar.slider('Skin Thickness (mm)', 0, 99, 20)
    insulin = st.sidebar.slider('Insulin (mu U/ml)', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 25.0, 0.1)
    diabetes_pedigree = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.372, 0.001)
    age = st.sidebar.slider('Age', 21, 81, 30)

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
    st.title("🏥 Diabetes Prediction System")
    st.subheader("AI-Powered Diabetes Risk Assessment")

    # Load model and scaler
    model, scaler = load_model_and_scaler()

    if model is None or scaler is None:
        st.error("Failed to load model. Please ensure the model is trained first.")
        st.info("Run `python src/train_model.py` to train the model.")
        return

    # Get user input
    features = get_user_input()

    # Display input features
    st.subheader("📋 Input Features")
    features_df = pd.DataFrame([features])
    st.dataframe(features_df, width='stretch')

    # Make prediction
    if st.button("🔍 Predict Diabetes Risk", type="primary"):
        with st.spinner("Analyzing your health data..."):
            prediction, probability = make_prediction(model, scaler, features)

        if prediction is not None:
            # Display prediction
            display_prediction(prediction, probability)

            # Risk factor analysis
            risk_factor_analysis(features, probability)

            # Generate and download report
            st.subheader("📄 Prediction Report")
            report = generate_report(features, prediction, probability)
            st.text_area("Report Preview", report, height=300)
            download_report(report)

    # Footer
    st.markdown("---")
    st.markdown("**Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice.")
    st.markdown("Built with ❤️ using Streamlit and Machine Learning")

if __name__ == "__main__":
    main()
