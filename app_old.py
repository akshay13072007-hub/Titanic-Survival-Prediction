import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("model.pkl")

# Page Title
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details below to predict survival.")

# User Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])

sex = st.selectbox("Sex", ["Male", "Female"])
sex = 0 if sex == "Male" else 1

age = st.slider("Age", 1, 80, 25)

sibsp = st.number_input("Number of Siblings/Spouses", 0, 10, 0)

parch = st.number_input("Number of Parents/Children", 0, 10, 0)

fare = st.number_input("Fare", 0.0, 600.0, 50.0)

embarked = st.selectbox("Embarked", ["S", "C", "Q"])

embarked = {"S": 0, "C": 1, "Q": 2}[embarked]

# Prediction
if st.button("Predict"):

    data = pd.DataFrame(
        [[pclass, sex, age, sibsp, parch, fare, embarked]],
        columns=[
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
        ],
    )

    prediction = model.predict(data)[0]

    if prediction == 1:
        st.success("✅ The passenger is likely to SURVIVE.")
    else:
        st.error("❌ The passenger is NOT likely to survive.")