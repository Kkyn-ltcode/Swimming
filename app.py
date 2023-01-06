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
    'pink_eye': 'Đau mắt đỏ',
    'diarrhea': 'Tiêu chảy',
    'otitis_externa': 'Viêm tai ngoài',
    'skin_rash': 'Da phát ban',
    'urinary_tract_infection': 'Nhiễm trùng đường tiểu'
}

with st.form('symtom_form'):
    st.markdown('# Symptoms')
    with st.expander(label='General', expanded=True):
        gender = st.radio(label='Giới tính', index=1,
                          options=['Male', 'Female'], horizontal=True)
        gender_encoded = 0 if gender == 'Female' else 1
        age = st.slider(label='Tuổi', min_value=5,
                        max_value=50, value=20, step=1)
        swimming_time = st.slider(label='Thời gian bơi', min_value=1,
                        max_value=24, value=1, step=1)
        swimming_time = swimming_time * 3600
        fever = st.radio(label='Sốt', index=1,
                               options=['Yes', 'No'], horizontal=True)
    cols1 = st.columns(2)
    with cols1[0]:
        with st.expander(label='Eye', expanded=True):
            conjunctiva_swelling = st.radio(label='Sưng giác mạc', index=1,
                                            options=['Yes', 'No'], horizontal=True)
            eye_itching = st.radio(label='Ngứa mắt', index=1,
                                   options=['Yes', 'No'], horizontal=True)
            eye_tearing = st.radio(label='Chảy nhiều nước mắt', index=1,
                                   options=['Yes', 'No'], horizontal=True)
    with cols1[1]:
        with st.expander(label='Urine', expanded=True):
            bloody_urine = st.radio(label='Đi tiểu ra máu', index=1,
                                    options=['Yes', 'No'], horizontal=True)
            pain_while_urinating = st.radio(label='Buốt khi tiểu', index=1,
                                            options=['Yes', 'No'], horizontal=True)
            frequent_urination = st.radio(label='Tiểu nhiều', index=1,
                                          options=['Yes', 'No'], horizontal=True)
    cols2 = st.columns(2)
    with cols2[0]:
        with st.expander(label='Ear', expanded=True):
            ear_pus = st.radio(label='Chảy mủ trong tai', index=1,
                               options=['Yes', 'No'], horizontal=True)
            ear_pain = st.radio(label='Đau tai', index=1,
                                options=['Yes', 'No'], horizontal=True)
            ear_itching = st.radio(label='Ngứa tai', index=1,
                                   options=['Yes', 'No'], horizontal=True)
            temporary_loss_hearing = st.radio(label='Mất thích giác tạm thời', index=1,
                                              options=['Yes', 'No'], horizontal=True)
    with cols2[1]:
        with st.expander(label='Skin', expanded=True):
            infection_skin = st.radio(label='Mủ rộp trên da', index=1,
                                            options=['Yes', 'No'], horizontal=True)
            skin_redness = st.radio(label='Da mẩn đỏ', index=1,
                                    options=['Yes', 'No'], horizontal=True)
            skin_itching = st.radio(label='Ngứa da', index=1,
                                    options=['Yes', 'No'], horizontal=True)
            pus_blisters_in_skin = st.radio(label='Bọng nước trên da', index=1,
                                            options=['Yes', 'No'], horizontal=True)

    with st.expander(label='Stomach', expanded=True):
        bloody_waste = st.radio(label='Đi ngoài ra máu', index=1,
                                options=['Yes', 'No'], horizontal=True)
        stomach_cramps = st.radio(label='Chướng bụng', index=1,
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
