import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
st.set_page_config(page_title="Wellness Tourism Predictor",page_icon="✈️")
st.title("Wellness Tourism Package - Purchase Prediction")
st.write("Enter customer details to estimate purchase likelihood.")
@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).parent/"deployment"/"model.joblib")
model=load_model()
with st.form("customer"):
    a,b=st.columns(2)
    with a:
        Age=st.number_input("Age",18,100,35)
        TypeofContact=st.selectbox("Type of Contact",["Self Enquiry","Company Invited"])
        CityTier=st.selectbox("City Tier",[1,2,3])
        DurationOfPitch=st.number_input("Duration of Pitch",min_value=0.0,value=10.0)
        Occupation=st.selectbox("Occupation",["Salaried","Small Business","Large Business","Free Lancer"])
        Gender=st.selectbox("Gender",["Male","Female","Fe Male"])
        NumberOfPersonVisiting=st.number_input("Number of People Visiting",1,20,2)
        NumberOfFollowups=st.number_input("Number of Follow-ups",0,20,3)
        ProductPitched=st.selectbox("Product Pitched",["Basic","Deluxe","Standard","Super Deluxe","King"])
    with b:
        PreferredPropertyStar=st.selectbox("Preferred Property Star",[3,4,5])
        MaritalStatus=st.selectbox("Marital Status",["Single","Married","Divorced","Unmarried"])
        NumberOfTrips=st.number_input("Number of Trips",0.0,50.0,3.0)
        Passport=st.selectbox("Has Passport?",[0,1])
        PitchSatisfactionScore=st.selectbox("Pitch Satisfaction Score",[1,2,3,4,5])
        OwnCar=st.selectbox("Owns a Car?",[0,1])
        NumberOfChildrenVisiting=st.number_input("Number of Children Visiting",0,10,0)
        Designation=st.selectbox("Designation",["Executive","Manager","Senior Manager","AVP","VP"])
        MonthlyIncome=st.number_input("Monthly Income",min_value=0.0,value=20000.0)
    submitted=st.form_submit_button("Predict")
if submitted:
    row=pd.DataFrame([locals()[k] for k in []])
    row=pd.DataFrame([{"Age":Age,"TypeofContact":TypeofContact,"CityTier":CityTier,"DurationOfPitch":DurationOfPitch,"Occupation":Occupation,"Gender":Gender,"NumberOfPersonVisiting":NumberOfPersonVisiting,"NumberOfFollowups":NumberOfFollowups,"ProductPitched":ProductPitched,"PreferredPropertyStar":PreferredPropertyStar,"MaritalStatus":MaritalStatus,"NumberOfTrips":NumberOfTrips,"Passport":Passport,"PitchSatisfactionScore":PitchSatisfactionScore,"OwnCar":OwnCar,"NumberOfChildrenVisiting":NumberOfChildrenVisiting,"Designation":Designation,"MonthlyIncome":MonthlyIncome}])
    p=float(model.predict_proba(row)[0,1])
    st.metric("Estimated purchase probability",f"{p:.1%}")
    st.success("Potential purchaser (class 1)" if p>=.5 else "Less likely to purchase (class 0)")
    st.caption("Model estimate only; not a guarantee of customer behaviour.")
