import pandas as pd
import streamlit as st
import joblib

model = joblib.load('Expressomodels.pkl')
data = pd.read_csv('expresso_processed.csv')

st.markdown("<h1 style = 'color: #0C2D57; text-align: center; font-family: geneva'>CHURN PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By ibrahim lawal</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

st.image('pngwing.com (3).png', use_column_width= True, )

st.header('Project Background Information',divider = True)
st.write("The primary objective of this project is to develop a sophisticated predictive model focused on forecasting churn rates within startup subscription services. Leveraging advanced machine learning techniques, the goal is to provide stakeholders with deep insights into the factors influencing subscription cancellations and customer attrition.Through the analysis of extensive datasets, the project aims to empower decision-makers with a comprehensive understanding of the dynamics impacting the success and sustainability of startup ventures. Problem statement is to  develop a predictive model to anticipate churn behavior among subscribers to startup services. Identify key factors contributing to churn within the context of startup subscription models. Provide actionable insights to stakeholders to mitigate churn and enhance customer retention strategies.")

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

# Input User Image 
st.sidebar.image('pngwing.com.png', caption = 'Welcome User')

# Apply space in the sidebar 
st.sidebar.markdown("<br>", unsafe_allow_html= True)
st.sidebar.markdown("<br>", unsafe_allow_html= True)

# Declare user Input variables 
st.sidebar.subheader('Input Variables', divider= True)
montant = st.sidebar.number_input('MONTANT', data['MONTANT'].min(), data['MONTANT'].max())
rev = st.sidebar.number_input('REVENUE', data['REVENUE'].min(), data['REVENUE'].max())
arpu = st.sidebar.number_input('ARPU_SEGMENT', data['ARPU_SEGMENT'].min(), data['ARPU_SEGMENT'].max())
freq = st.sidebar.number_input('FREQUENCE', data['FREQUENCE'].min(), data['FREQUENCE'].max())
data_vol = st.sidebar.number_input('DATA_VOLUME', data['DATA_VOLUME'].min(), data['DATA_VOLUME'].max())
onnet = st.sidebar.number_input('ON_NET', data['ON_NET'].min(), data['ON_NET'].max())
reg = st.sidebar.number_input('REGULARITY', data['REGULARITY'].min(), data['REGULARITY'].max())
Ten = st.sidebar.selectbox('TENURE', data['TENURE'].unique())
mrg = st.sidebar.selectbox('MRG', ["NO"])

#sel_cols = ['MONTANT', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'CHURN', 'REGULARITY', 'TENURE', 'MRG']



# display the users input
input_var = pd.DataFrame()
input_var['MONTANT'] = [montant]
input_var['REVENUE'] = [rev]
input_var['ARPU_SEGMENT'] = [arpu]
input_var['FREQUENCE'] = [freq]
input_var['DATA_VOLUME'] = [data_vol]
input_var['ON_NET'] = [onnet]
input_var['REGULARITY'] = [reg]
input_var['TENURE'] = [Ten]
input_var['MRG'] = [mrg]


st.markdown("<br>", unsafe_allow_html= True)
# display the users input variable 
st.subheader('Users Input Variables', divider= True)
st.dataframe(input_var)

arpu = joblib.load('ARPU_SEGMENT_scaler.pkl')
data_vol = joblib.load('DATA_VOLUME_scaler.pkl')
montant = joblib.load('MONTANT_scaler.pkl')
mrg = joblib.load('MRG_encoder.pkl')
revenue = joblib.load('REVENUE_scaler.pkl')
tenure = joblib.load('TENURE_encoder.pkl')


# transform the users input with the imported scalers 
input_var['ARPU_SEGMENT'] = arpu.transform(input_var[['ARPU_SEGMENT']])
input_var['DATA_VOLUME'] = data_vol.transform(input_var[['DATA_VOLUME']])
input_var['MONTANT'] = montant.transform(input_var[['MONTANT']])
input_var['MRG'] = mrg.transform(input_var[['MRG']])
input_var['REVENUE'] = revenue.transform(input_var[['REVENUE']])
input_var['TENURE'] = tenure.transform(input_var[['TENURE']])

model = joblib.load('Expressomodels.pkl')
predicted = model.predict(input_var)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

if st.button('Predict Churn'):
    if predicted == 0:
        st.failure('Customer Has CHURNED')
    else:
        st.success('Customer Is With Us')