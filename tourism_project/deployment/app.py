import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="nmvvskb/best_tourism_model", filename="best_tourism_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Visit with us tourism Prediction
st.title("Wellness Tourism Package Prediction App")
st.write("This application predicts whether a customer is likely to purchase the **Wellness Tourism Package** offered by **Visit with Us**.")
st.write("Enter the customer details in the form below and the system will estimate whether the customer is likely to purchase the Wellness Tourism Package")

# Collect user input
# Numeric Inputs
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration Of Pitch", min_value=1.0, value=150.0)
NumberOfPersonVisiting = st.number_input("Number Of Person Visiting", min_value=1, value=2)
NumberOfFollowups = st.number_input("Number Of Followups", min_value=0.0, value=1.0)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
NumberOfTrips = st.number_input("Number Of Trips", min_value=0.0, value=1.0)
Passport = st.selectbox("Passport", [0, 1])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
OwnCar = st.selectbox("Own Car", [0, 1])
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", min_value=0, value=0)
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=300000.0)

#Categorical 
TypeofContact = st.selectbox("Type Of Contact",["Company Invited", "Self Enquiry"])
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female" ,"Fe Male"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard" ,"Deluxe", "Super Deluxe", "King"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married" ,"Divorced" , "Unmarried"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase a wellness package" if prediction == 1 else "not purchase a wellness package"
    st.write(f"Based on the information provided, the customer is likely to {result}.")

