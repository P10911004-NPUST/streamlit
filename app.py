# streamlit run pycode\app.py

import os
import pandas as pd
import numpy as np

from pycaret import regression as reg
from pycaret import classification as clf

import plotly.express as px
import streamlit as st

# st.markdown("""
# <style>
# @font-face {
#     font-family: 'A';
#     font-style: normal;
#     font-weight: 400;
#     src: url(https://fonts.gstatic.com/s/tangerine/v12/IurY6Y5j_oScZZow4VOxCZZM.woff2) format('woff2');
#     unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+2074, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD;
# }

# html, body, [class*="css"] {
#     font-family: 'serif';
#     font-size: 20px;
#     }
#     </style>

#     """,
#         unsafe_allow_html=True,
# )

mod_list = os.listdir("./models")
reg_mod = [i for i in mod_list if i.endswith('.pkl') and i.startswith('reg')][0]
clf_mod = [i for i in mod_list if i.endswith('.pkl') and i.startswith('clf')][0]
mape_file = [i for i in mod_list if i.startswith('mape')][0]

with open(f"./models/{mape_file}") as f:
    mape = np.float64(f.readline())

st.title("Insurance charges estimator")
st.write("This is a trial")

age = st.number_input('Age', value=20)
sex = st.selectbox("Sex", ("male", "female"))
bmi = st.number_input('Insert BMI', value=20)
children = st.radio("Children", ("yes", "no"))
region = st.multiselect(
    label='Region',
    options=['southeast', 'southwest', 'northeast', 'northwest'],
    default=['southeast']
)
# charges = st.number_input("Charges", value=8888.888)

reg_x = pd.DataFrame({
    'smoker': 'yes',
    'age': age,
    'sex': sex,
    'bmi': bmi,
    'children': children,
    'southeast': str(np.where('southeast' in region, 'yes', 'no')),
    'southwest': str(np.where('southwest' in region, 'yes', 'no')),
    'northeast': str(np.where('northeast' in region, 'yes', 'no')),
    'northwest': str(np.where('northwest' in region, 'yes', 'no'))
}, index=[0])

st.table(reg_x)

if st.button('Estimate'):
    regressor = reg.load_model(os.path.join("./models", reg_mod.split('.')[0]))
    classifier = clf.load_model(os.path.join("./models", clf_mod.split('.')[0]))

    nx = 100

    charges = reg.predict_model(regressor, reg_x)
    charges = int(charges['prediction_label'][0])
    min_charges = int(charges - charges * mape)
    max_charges = int(charges + charges * mape)
    charges_range = np.linspace(start=min_charges, stop=max_charges, num=nx)
    charges_range = np.round(charges_range, 0)

    clf_x = pd.DataFrame({
        'charges': charges_range,
        'age': np.repeat(age, nx),
        'sex': np.repeat(sex, nx),
        'bmi': np.repeat(bmi, nx),
        'children': np.repeat(children, nx),
        'southeast': np.repeat(reg_x.southeast[0], nx),
        'southwest': np.repeat(reg_x.southwest[0], nx),
        'northeast': np.repeat(reg_x.northeast[0], nx),
        'northwest': np.repeat(reg_x.northwest[0], nx),
    })

    smoker = clf.predict_model(classifier, clf_x)
    smoker_prob = (1 - smoker["prediction_score"]) * 100

    st.write(f"Charges range: {min_charges} ~ {max_charges}")

    fig = px.area(
        x=charges_range, 
        y=smoker_prob,
        color_discrete_sequence=["#F63366"],
        template='simple_white',
        labels={
            "x": "Charges 報價 (NTD)",
            "y": "Deal rate 成交機率 (%)"
        })
    fig.update_layout(
        font=dict(
            family="serif",
            size=16,
            
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# streamlit run pycode\app.py

