import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Load model from Hugging Face Model Hub
model_path = hf_hub_download(repo_id="your-username/visit-with-us-model", filename="best_model.pkl")
model = joblib.load(model_path)

st.title("Wellness Tourism Package Purchase Predictor")

st.markdown("Enter customer details to predict the likelihood of purchase.")

# Input form
with st.form("input_form"):
    age = st.number_input("Age", min_value=18, max_value=100)
    typeof_contact = st.selectbox("Type of Contact", list(contact_map.keys()))
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", list(occupation_map.keys()))
    gender = st.selectbox("Gender", list(gender_map.keys()))
    marital_status = st.selectbox("Marital Status", list(marital_map.keys()))
    designation = st.selectbox("Designation", list(designation_map.keys()))
    number_of_trips = st.slider("Number of Trips per Year", 0, 20, 1)
    passport = st.selectbox("Has Passport", [0, 1])
    own_car = st.selectbox("Owns Car", [0, 1])
    number_of_children = st.slider("Number of Children Visiting", 0, 5, 0)
    pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    product_pitched = st.selectbox("Product Pitched", list(product_map.keys()))
    followups = st.slider("Number of Follow-ups", 0, 10, 1)
    duration = st.slider("Duration of Pitch (minutes)", 0, 60, 10)

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": contact_map[typeof_contact],
        "CityTier": city_tier,
        "Occupation": occupation_map[occupation],
        "Gender": gender_map[gender],
        "MaritalStatus": marital_map[marital_status],
        "Designation": designation_map[designation],
        "NumberOfTrips": number_of_trips,
        "Passport": passport,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": number_of_children,
        "PitchSatisfactionScore": pitch_score,
        "ProductPitched": product_map[product_pitched],
        "NumberOfFollowups": followups,
        "DurationOfPitch": duration
    }])
        prediction = model.predict(input_df)[0]
        st.success(f"Prediction: {'Will Purchase' if prediction == 1 else 'Will Not Purchase'}")
