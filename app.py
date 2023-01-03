import streamlit as st
import numpy as np
import pandas as pd
import model
import pickle
import json

st.set_page_config(
    page_title="Swimming Diseases", page_icon="random", initial_sidebar_state="expanded", layout="wide"
)

diseases = {
    'pink_eye': 'Pink Eye',
    'diarrhea': 'Diarrhea',
    'otitis_externa': 'Otitus Externa',
    'skin_rash': 'Skin Rash',
    'urinary_tract_infection': 'Urinary Tract Infection'
}

with st.form('symtom_form'):
    st.markdown('# Symptoms')
    with st.expander(label='General', expanded=True):
        gender = st.radio(label='gender', index=1,
                          options=['Male', 'Female'], horizontal=True)
        gender_encoded = 0 if gender == 'Female' else 1
        age = st.slider(label='age', min_value=5,
                        max_value=50, value=20, step=1)
        swimming_time = st.number_input(
            label="swimming_time", min_value=0, max_value=24)
        swimming_time = swimming_time * 3600
        fever = st.radio(label='fever', index=1,
                               options=['Yes', 'No'], horizontal=True)
    cols1 = st.columns(2)
    with cols1[0]:
        with st.expander(label='Eye', expanded=True):
            conjunctiva_swelling = st.radio(label='conjunctiva_swelling', index=1,
                                            options=['Yes', 'No'], horizontal=True)
            eye_itching = st.radio(label='eye_itching', index=1,
                                   options=['Yes', 'No'], horizontal=True)
            eye_tearing = st.radio(label='eye_tearing', index=1,
                                   options=['Yes', 'No'], horizontal=True)
    with cols1[1]:
        with st.expander(label='Urine', expanded=True):
            bloody_urine = st.radio(label='bloody_urine', index=1,
                                    options=['Yes', 'No'], horizontal=True)
            pain_while_urinating = st.radio(label='pain_while_urinating', index=1,
                                            options=['Yes', 'No'], horizontal=True)
            frequent_urination = st.radio(label='frequent_urination', index=1,
                                          options=['Yes', 'No'], horizontal=True)
    cols2 = st.columns(2)
    with cols2[0]:
        with st.expander(label='Ear', expanded=True):
            ear_pus = st.radio(label='ear_pus', index=1,
                               options=['Yes', 'No'], horizontal=True)
            ear_pain = st.radio(label='ear_pain', index=1,
                                options=['Yes', 'No'], horizontal=True)
            ear_itching = st.radio(label='ear_itching', index=1,
                                   options=['Yes', 'No'], horizontal=True)
            temporary_loss_hearing = st.radio(label='temporary_loss_hearing', index=1,
                                              options=['Yes', 'No'], horizontal=True)
    with cols2[1]:
        with st.expander(label='Skin', expanded=True):
            infection_skin = st.radio(label='infection_skin', index=1,
                                            options=['Yes', 'No'], horizontal=True)
            skin_redness = st.radio(label='skin_redness', index=1,
                                    options=['Yes', 'No'], horizontal=True)
            skin_itching = st.radio(label='skin_itching', index=1,
                                    options=['Yes', 'No'], horizontal=True)
            pus_blisters_in_skin = st.radio(label='pus_blisters_in_skin', index=1,
                                            options=['Yes', 'No'], horizontal=True)

    with st.expander(label='Stomach', expanded=True):
        bloody_waste = st.radio(label='bloody_waste', index=1,
                                options=['Yes', 'No'], horizontal=True)
        stomach_cramps = st.radio(label='stomach_cramps', index=1,
                                  options=['Yes', 'No'], horizontal=True)
    submited = st.form_submit_button(label='Submit')
    if submited:
        input_data = [
            age,
            swimming_time,
            gender_encoded,
            conjunctiva_swelling,
            eye_itching,
            eye_tearing,
            bloody_waste,
            stomach_cramps,
            fever,
            ear_pus,
            ear_pain,
            ear_itching,
            temporary_loss_hearing,
            infection_skin,
            skin_redness,
            skin_itching,
            pus_blisters_in_skin,
            bloody_urine,
            pain_while_urinating,
            frequent_urination
        ]
        with open('config.json') as json_file:
            config = json.load(json_file)
        input_data_encoded = pd.DataFrame(
            [[0 if i == 'No' else 1 if i == 'Yes' else i for i in input_data]],
            columns=config['feature_name'])

        results = {}
        for disease in diseases.keys():
            model_name = f'model/{disease}.sav'
            model = pickle.load(open(model_name, "rb"))
            prediction = model.predict(input_data_encoded)
            results[disease] = prediction[0]
        if sum(results.values()) == 0:
            st.write('Normal')
        else:
            for disease_name, prediction in results.items():
                if prediction == 1:
                    st.write(f'Disease: {disease_name}')
                    st.write(config['advise'][disease_name])
