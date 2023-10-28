import streamlit as st
import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from PIL import Image
import pickle

image = Image.open("D:/EHTP_Data_Engineering/Data engineering/Master Degree Project/Officia_last_Project/Midas_Logo.png")

st.set_page_config(layout='wide', initial_sidebar_state='expanded', page_icon = image)
st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
st.markdown("""
  <style>
    .css-1vq4p4l.e1fqkh3o4 {
      margin-top: -105px;
    }
  </style>
""", unsafe_allow_html=True)

st.sidebar.image(image, width =250)

st.markdown("""# MIDAS : AGREA
## Behaviour Prediction App
This app predicts *Customers Behaviour* """)


dg = st.sidebar.file_uploader("Upload Customer File",type=["csv"])
if dg is not None:
    dg = pd.read_csv(dg,index_col='Id_Client')
    dg=dg[dg['120_dép']==0]
    st.write(dg[:5])
    fig = px.scatter(
    data_frame=dg, x="Proba_Rfr", y="MT_Prédi_Rfr", title="Sample Data",trendline="ols", color_discrete_sequence=['darkblue'],)
    fig.update_layout(plot_bgcolor='lightgrey')
    left, right = st.columns(2) 
    with right: 
        st.subheader("Distribution des résultats prédits")
        st.plotly_chart(fig)    
    fig1 = px.histogram(dg, x="Recency")
    fig1.update_layout(autosize=False,width=500,height=200,)
    fig2 = px.histogram(dg, x="Frequency")
    fig2.update_layout(autosize=False,width=500,height=200,)
    fig3 = px.histogram(dg, x="Prix_Total")
    fig3.update_layout(autosize=False,width=500,height=200,)
    with left:
        st.subheader("Distribution du RFM")
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)
        st.plotly_chart(fig3)

def user_input_features():
    Recency = st.sidebar.slider("Select Recency",1,500,0)
    Frequency = st.sidebar.slider("Select Frequency",1,500,0)
    Monetary = st.sidebar.slider("Select Monetary",1,500000,0)
    Prix_Moyenne = st.sidebar.slider("Select Mean_Price",1,250000,0)
    data = {'Recency': Recency,
            'Frequency': Frequency,
            'Prix_Total': Monetary,
            'Prix_Moyenne':Prix_Moyenne}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()




st.subheader('User Input parameters')
st.write(df)

model_cl=pickle.load(open("D:/EHTP_Data_Engineering/Data engineering/Master Degree Project/Officia_last_Project/rf_cl.pkl", "rb"))
model_reg=pickle.load(open("D:/EHTP_Data_Engineering/Data engineering/Master Degree Project/Officia_last_Project/rf_reg.pkl", "rb"))
prediction = model_reg.predict(df)
prediction_proba = model_cl.predict_proba(df)


st.subheader('Classe Achat Oui/Non')
st.write(pd.DataFrame(model_cl.classes_))
left, right = st.columns(2)  
with right:
    st.subheader('Montant de dépense Prédit')
    st.write(prediction.round(2))
with left:     
    st.subheader('Probabilité d\'achat')
    st.write(prediction_proba.round(2))