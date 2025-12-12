import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------
# Load pre-trained model from Hugging Face Model Hub
# ---------------------------------------------------------

HF_REPO_ID = "JefferyMendis/tourism-package-model"     # Update if needed
MODEL_FILENAME = "model.joblib"

st.write("Loading model...")

try:
    # Downloads model to local cache if not already present
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model",
        token=os.getenv("HF_TOKEN")  # recommended: set at runtime
    )
    model = joblib.load(model_path)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ---------------------------------------------------------
# Streamlit App UI
# ---------------------------------------------------------

st.title("🏖️ Tourism Package Purchase Prediction")
st.write("Predict whether a customer is likely to buy the **Wellness Tourism Package**.")

st.subheader("🧍 Customer Information")

# -------- Numeric Inputs -------
age = st.number_input("Age", min_value=18, max_value=100, value=30)
duration = st.number_input("Duration of Sales Pitch (minutes)", 0.0, 100.0, 10.0)
visiting = st.number_input("Number of Persons Visiting", 1, 10, 2)
followups = st.number_input("Number of Follow-ups", 0, 20, 3)
trips = st.number_input("Number of Trips per Year", 0, 20, 2)
score = st.number_input("Pitch Satisfaction Score", 1, 5, 3)
children = st.number_input("Number of Children Visiting", 0, 10, 0)
income = st.number_input("Monthly Income (₹)", 0.0, 200000.0, 25000.0)

# -------- Categorical Inputs -------
st.subheader("📊 Customer Preferences & Attributes")

contact = st.selectbox("Type of Contact", ['Self Inquiry', 'Company Invited'])
occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
gender = st.selectbox("Gender", ['Female', 'Male'])
product = st.selectbox("Product Pitched", ['Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'King'])
marital = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced', 'Unmarried'])
designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])
tier = st.selectbox("City Tier", [1, 2, 3])
star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
passport = st.selectbox("Has Passport?", [0, 1])
car = st.selectbox("Owns a Car?", [0, 1])

# ---------------------------------------------------------
# Prepare Input DataFrame (must match training schema)
# ---------------------------------------------------------
input_data = pd.DataFrame([{
    'Age': age,
    'DurationOfPitch': duration,
    'NumberOfPersonVisiting': visiting,
    'NumberOfFollowups': followups,
    'NumberOfTrips': trips,
    'PitchSatisfactionScore': score,
    'NumberOfChildrenVisiting': children,
    'MonthlyIncome': income,
    'TypeofContact': contact,
    'Occupation': occupation,
    'Gender': gender,
    'ProductPitched': product,
    'MaritalStatus': marital,
    'Designation': designation,
    'CityTier': tier,
    'PreferredPropertyStar': star,
    'Passport': passport,
    'OwnCar': car
}])

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if st.button("🔮 Predict Purchase Outcome"):
    try:
        prediction = model.predict(input_data)[0]
        proba = None

        # Try getting probability (if pipeline supports it)
        try:
            proba = model.predict_proba(input_data)[0][1]
        except:
            pass

        if prediction == 1:
            st.success("Result: ✅ **Customer Likely to Purchase**")
        else:
            st.error("Result: ❌ **Customer Unlikely to Purchase**")

        if proba is not None:
            st.info(f"**Purchase Probability:** {proba:.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
