import pickle
import streamlit as st

st.set_page_config(page_title="Predicción diabetes")

# Cargar modelo
model_path = "models/diab_tree_classifier_crit-entro_maxdepth-5_minleaf-2_minsplit10_42.sav"
model = pickle.load(open(model_path, "rb"))

class_dict = {0: "No diabético", 
              1: "Diabético"}

st.title("Predicción diabético")
st.caption(f"Loaded model: {model_path}")
st.divider()

# Inputs
Pregnancies = st.slider("Pregnancies", min_value=0, max_value=17, value=1, step=1)
Glucose = st.slider("Glucose", min_value=0, max_value=200, value=120, step=1)
BloodPressure = st.slider("Blood Pressure", min_value=0, max_value=122, value=70, step=1)
SkinThickness = st.slider("Skin Thickness", min_value=0, max_value=99, value=20, step=1)
Insulin = st.slider("Insulin", min_value=0, max_value=846, value=80, step=1)
BMI = st.slider("BMI", min_value=0.0, max_value=67.1, value=30.0, step=0.1)
DiabetesPedigreeFunction = st.slider("Diabetes Pedigree Function", min_value=0.078, max_value=2.42, value=0.47, step=0.01)
Age = st.slider("Age", min_value=21, max_value=81, value=33, step=1)

st.divider()

if st.button("Predict"):
    X = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
    pred = int(model.predict(X)[0])
    st.divider()
    st.subheader("Result")
    st.write(class_dict[pred])
    st.divider()
 

