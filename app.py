import numpy as np
import joblib
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺")


@st.cache_resource
def load_artifacts():
    model = load_model("diabetes_model.keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

st.title("Diabetes Risk Predictor")

YES_NO = {"No": 0, "Yes": 1}

with st.form("input_form"):
    c1, c2 = st.columns(2)

    # Order below matches the training columns exactly
    HighBP = YES_NO[c1.selectbox("High blood pressure", list(YES_NO))]
    HighChol = YES_NO[c2.selectbox("High cholesterol", list(YES_NO))]
    CholCheck = YES_NO[c1.selectbox("Cholesterol check in last 5 years", list(YES_NO))]
    BMI = c2.number_input("BMI", 10.0, 100.0, 25.0, 0.1)
    Smoker = YES_NO[c1.selectbox("Smoked 100+ cigarettes in life", list(YES_NO))]
    Stroke = YES_NO[c2.selectbox("Ever had a stroke", list(YES_NO))]
    HeartDiseaseorAttack = YES_NO[c1.selectbox("Heart disease / heart attack", list(YES_NO))]
    PhysActivity = YES_NO[c2.selectbox("Physical activity in last 30 days", list(YES_NO), index=1)]
    Fruits = YES_NO[c1.selectbox("Eats fruit daily", list(YES_NO), index=1)]
    Veggies = YES_NO[c2.selectbox("Eats vegetables daily", list(YES_NO), index=1)]
    HvyAlcoholConsump = YES_NO[c1.selectbox("Heavy alcohol consumption", list(YES_NO))]
    AnyHealthcare = YES_NO[c2.selectbox("Has healthcare coverage", list(YES_NO), index=1)]
    NoDocbcCost = YES_NO[c1.selectbox("Skipped doctor because of cost", list(YES_NO))]
    GenHlth = c2.number_input("General health (1 excellent - 5 poor)", 1, 5, 3)
    MentHlth = c1.number_input("Bad mental health days (last 30)", 0, 30, 0)
    PhysHlth = c2.number_input("Bad physical health days (last 30)", 0, 30, 0)
    DiffWalk = YES_NO[c1.selectbox("Difficulty walking / climbing stairs", list(YES_NO))]
    Sex = {"Female": 0, "Male": 1}[c2.selectbox("Sex", ["Female", "Male"])]
    Age = c1.number_input("Age group (1 = 18-24 ... 13 = 80+)", 1, 13, 7)
    Education = c2.number_input("Education level (1 - 6)", 1, 6, 4)
    Income = c1.number_input("Income level (1 - 8)", 1, 8, 5)

    submitted = st.form_submit_button("Predict")

if submitted:
    row = np.array([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
                     HeartDiseaseorAttack, PhysActivity, Fruits, Veggies,
                     HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth,
                     MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income]],
                   dtype=float)
    prob = model.predict(scaler.transform(row), verbose=0)[0]
    labels = ["No Diabetes", "Diabetes"]
    k = int(np.argmax(prob))

    if k == 1:
        st.error(f"Prediction: {labels[k]}")
    else:
        st.success(f"Prediction: {labels[k]}")
    for name, p in zip(labels, prob):
        st.write(f"{name}: {p:.1%}")
        st.progress(float(p))
