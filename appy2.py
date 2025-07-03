import streamlit
import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
filename = "finalized_model.sav"
model = joblib.load(filename)
data = pd.read_csv("Financial_inclusion_dataset.csv")
colums = ['year', 'household_size','gender_of_respondent','country', 'bank_account', 'location_type', 'cellphone_access' , 'relationship_with_head', 'marital_status', 'education_level', 'job_type']
colums_numerique =  [ 'year', 'household_size','gender_of_respondent' ] 
data = data[colums]
label_enc = LabelEncoder()
col_val = {}
for col in ['country', 'bank_account', 'location_type', 'cellphone_access' , 'relationship_with_head', 'marital_status', 'education_level', 'job_type']:
    data[col] = label_enc.fit_transform(data[col])
    col_val[col] = st.radio(f"Select {col} ", tuple(label_enc.classes_)) 
for col in colums_numerique:
    data[col] = label_enc.fit_transform(data[col])
    col_val[col] = st.number_input(f"Entrer la valeur pour {col}", value=0.0)
# Prédiction lorsque le bouton est cliqué
if st.button("Prédire"):
    input_df = pd.DataFrame([col_val])

    try:
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)[0][1]

        st.success(f"Résultat: {'bank_account' if prediction[0] == 1 else 'Pas de bank_account'}")
        st.info(f"Probabilité: {proba:.0%}")

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")    