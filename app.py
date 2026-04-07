import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("credits_model.pkl", "rb"))

st.title(" Credit Risk Prediction")
st.caption("Model: Logistic Regression (Best ROC-AUC)")

st.write("Enter applicant details:")

income = st.number_input("Total Income", min_value=0.0, value=100000.0)
credit = st.number_input("Credit Amount", min_value=0.0, value=500000.0)
annuity = st.number_input("Loan Annuity", min_value=0.0, value=20000.0)

age = st.number_input("Age (in years)", min_value=18, max_value=100, value=30)
employment = st.number_input("Years Employed", min_value=0, value=5)

ext2 = st.number_input("External Score 2 (0–1)", min_value=0.0, max_value=1.0, value=0.5)
ext3 = st.number_input("External Score 3 (0–1)", min_value=0.0, max_value=1.0, value=0.5)

income_type = st.selectbox(
    "Income Type",
    ["Working", "Commercial associate", "State servant", "Pensioner"]
)

education = st.selectbox(
    "Education Type",
    ["Secondary / secondary special", "Higher education", "Lower secondary"]
)

input_dict = {
    "DAYS_EMPLOYED": -employment * 365,
    "DAYS_BIRTH": -age * 365,
    "AMT_CREDIT": credit,
    "AMT_INCOME_TOTAL": income,
    "AMT_ANNUITY": annuity,
    "EXT_SOURCE_2": ext2,
    "EXT_SOURCE_3": ext3,
    "NAME_INCOME_TYPE": income_type,
    "NAME_EDUCATION_TYPE": education
}

input_df = pd.DataFrame([input_dict])

if st.button("Predict"):
    prob = model.predict_proba(input_df)[0][1]

    st.subheader(f"Risk Score: {prob:.2f} ({prob*100:.1f}%)")

    if prob > 0.25:
        st.error(" High Risk - Not Eligible")
    else:
        st.success("Low Risk - Eligible")

    if prob > 0.6:
        st.warning("Very High Risk ")
    elif prob > 0.25:
        st.info("Moderate Risk ")
    else:
        st.success("Low Risk ")