# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:21:08 2022

@author: Chandrasekaran
"""


import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from IPython.core.pylabtools import figsize
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

#Styling
st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 90%;
        padding-top: 5rem;
        padding-right: 5rem;
        padding-left: 5rem;
        padding-bottom: 5rem;
    }}
    img{{
    	max-width:40%;
    	margin-bottom:40px;
    }}
</style>
""",
        unsafe_allow_html=True,
    )
#----General
valid_molecule = True
loaded_molecule = None
selection = None
submit = None
#-----sidebar
page = st.sidebar.selectbox('Select Options', ["Data", "Visualization", "Model Analysis"])
st.sidebar.markdown("""---""")
st.sidebar.write("Created by [Chandrasekaran](https://www.linkedin.com/in/chandra-sekaran-52b160b3/)")
#st.sidebar.image("PIC CHAN.jpeg", width=100)

header = st.container()
#st.image('TimeSeries.jpg')
st.title("An Interactive Dashboard for Time-Series")
st.subheader("------------------------")
st.write("-------")
data = pd.read_csv('all.csv')
global numeric_columns
if page == "Data":
	
	stats = st.container()
		
	with stats:
		st.header('We have taken the below dataset for this analysis')
		data = pd.read_csv('all.csv')
		st.write(data)

#Visalization
elif page == "Visualization":

	vis = st.container()
	with vis:
		st.header('Data Visualization tool')
		st.write('This is to understand the data more and in-depth analysis')
		numeric_columns = list(data.select_dtypes(['float','int']).columns)
		new = st.selectbox('Which plot would you like to see?',["Histogram", "Lineplot", "Maps", "Piechart"])	
		
		if new == "Histogram":
			st.write('Please select the features:')
			try:
				x_values = st.selectbox('X-axis', options = numeric_columns)
				y_values = st.selectbox('Y-axis', options = numeric_columns)
				plot = px.histogram(data_frame=data, x=x_values, y=y_values)
				st.plotly_chart(plot)
			except Exception as e:
				print(e)
		elif new == "Maps":
			st.write('Please select the features:')
			try:
				px.set_mapbox_access_token("pk.eyJ1IjoidmltYWwxMjM0IiwiYSI6ImNsMXN2dGlmMDI3cjgzY28yaXNxZWR3ZnEifQ.gva5d-xA6tC-y191P8wRPA")
				df = data.copy()
				fig = px.scatter_mapbox(data_frame=df, lat="lat", lon="lon",zoom =15,hover_data=["outgoing_site_id","Traffic", "Call Dropped"], size_max=100)
				st.plotly_chart(fig)
	
			except Exception as e:
				print(e)
		elif new == "Lineplot":
			st.write('Please select the features:')
			try:
				x_values = st.selectbox('X-axis', options = numeric_columns)
				y_values = st.selectbox('Y-axis', options = numeric_columns)
				plot = px.line(data_frame=data, x=x_values, y=y_values)
				st.plotly_chart(plot)
			except Exception as e:
				print(e)
		else:
			st.write('Please select the features:')
			try:
				x_values = st.selectbox('X-axis', options = numeric_columns)
				y_values = st.selectbox('Y-axis', options = numeric_columns)
				plot = px.pie(data_frame=data, names=x_values, values=y_values)
				st.plotly_chart(plot)
			except Exception as e:
				print(e)
		
			
#Model_training
else:
	with st.container():
		def build_model(data):
			x = data.drop(columns = ['Call Dropped'])
			y = data[['Call Dropped']]

			

			
		
		
		
		





