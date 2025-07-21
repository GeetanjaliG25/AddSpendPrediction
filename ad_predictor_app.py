import streamlit as st
import pandas as pd
import joblib 

model = joblib.load('add_model.pkl')  

st.title("Add Spend → Sales Predictor")

tv = st.number_input("Total Ad Budget")
social = st.number_input("Social Media Spend")
influencer = st.number_input("Influencer Spend")

if st.button("Predict"):
    input_df = pd.DataFrame([[tv, social, influencer]], columns=['Total_Ad_Budget', 'Social_Spend', 'Influencer_Spend'])
    result = model.predict(input_df)
    #st.success(f"Predicted Sales: {result[0]:,.2f}")
    st.success(f"Predicted Sales: {float(result[0]):,.2f}")



