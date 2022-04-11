import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from IPython.core.pylabtools import figsize
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

st.set_page_config(page_title='The Machine Learning App',
    layout='wide')

#Model build

#Main 
st.title('An Interactive Machine Learning Dashboard')
st.subheader('This Dashboard was created by Chandrasekaran, Vimalesh Raja')
st.write('In this implementation, the *XGBRegressor()* function is used in this app to build a regression model using the XGBoost algorithm.')



#Page navigation
page = st.sidebar.selectbox('Select Analysis', ["Visualization", "Forecasting"])
# Displays the dataset
with st.sidebar.header('Please Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
st.subheader('Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.set_index('Datetime')
    df['Start_Time_MM_DD_YYYY'] = pd.to_datetime(df.Start_Time_MM_DD_YYYY , format = '%Y%m%d')
    st.markdown('**Glimpse of dataset**')
    st.write(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')

if page == "Visualization":
		st.header('Data Visualization tool')
		st.write('This is to understand the data more and in-depth analysis')
		numeric_columns = list(df.select_dtypes(['float','int']).columns)
		new = st.selectbox('Which plot would you like to see?',["Histogram", "Lineplot", "Piechart"])	
		
		if new == "Histogram":
			st.write('Please select the features:')
			try:
				x_values = st.selectbox('X-axis', options = numeric_columns)
				y_values = st.selectbox('Y-axis', options = numeric_columns)
				plot = px.histogram(data_frame=df, x=x_values, y=y_values, color='day')
				st.plotly_chart(plot)
			except Exception as e:
				print(e)
		elif new == "Lineplot":
			st.write('Please select the features:')
			try:
				x_values = st.selectbox('X-axis', options = numeric_columns)
				y_values = st.selectbox('Y-axis', options = numeric_columns)
				plot = px.line(data_frame=df, x=x_values, y=y_values, title='Comparisons', color='day', symbol='drop%')
				st.plotly_chart(plot)
			except Exception as e:
				print(e)
		else:
			st.write('Please select the features:')
			try:
				x_values = st.selectbox('X-axis', options = numeric_columns)
				y_values = st.selectbox('Y-axis', options = numeric_columns)
				plot = px.pie(data_frame=df, names=x_values, values=y_values, title='Pie Chart Comparisons')
				st.plotly_chart(plot)
			except Exception as e:
				print(e)

else:
    print('Under Development')