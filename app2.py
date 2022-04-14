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


st.set_page_config(layout="wide")
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

st.write(
    '<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

#--

#----General
valid_molecule = True
loaded_molecule = None
selection = None
submit = None
#-----sidebar
page = st.sidebar.selectbox(
    'Select Options', ["Data", "Visualization", "Forecasting"])
st.sidebar.markdown("""---""")

header = st.container()
st.title("An Interactive Dashboard for Time-Series")
st.subheader("------------------------")
st.write("-------")
data = pd.read_csv('FinalDataset.csv')

######
def build_model(df):
	df=df.reset_index()
	df.index=df.Datetime
	df=df.drop(columns=['Datetime'])
	x=df.drop(columns = ['Call Dropped'])
	y= df[['Call Dropped']]

	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25,random_state = 0, shuffle = False)
	model = XGBRegressor(objective = 'reg:squarederror',n_estimators =49)
	model.fit(x_train,y_train)

	y_train_pred=model.predict(x_train)
	prediction1= model.predict(x_test)
	return x_train, x_test, y_train, y_test, y_train_pred,prediction1



#### End of ML


global numeric_columns
if page == "Data":

	stats = st.container()
	with stats:
		st.header('Glimpse of the Data')
		st.write(data)

#Visalization
elif page == "Visualization":
	vis = st.container()
	with vis:
		col1, col2 = st.columns(2)
		col1.header('Data Visualization tool')
		col1.write('This is to understand the data more and in-depth analysis')
		numeric_columns = list(data.columns)
		new = col1.selectbox('Which plot would you like to see?', [
		                     "Histogram", "Lineplot", "Maps", "Piechart"])

		if new == "Histogram":
			col1.write('Please select the features:')
			siteid = col1.radio('Select Site ID', data['outgoing_site_id'].unique())
			df = data.loc[data['outgoing_site_id'] == siteid].reset_index()
			try:
				x_values = col1.selectbox('X-axis', options=numeric_columns)
				y_values = col1.selectbox('Y-axis', options=numeric_columns)
				plot = px.histogram(data_frame=data, x=x_values, y=y_values, color='weather')
				with col2:
					st.write('Hist plot for the Site ID:', siteid)
					st.plotly_chart(plot)
			except Exception as e:
				print(e)
		elif new == "Maps":
			st.write('Please select the features:')
			try:
				px.set_mapbox_access_token(
				    "pk.eyJ1IjoidmltYWwxMjM0IiwiYSI6ImNsMXN2dGlmMDI3cjgzY28yaXNxZWR3ZnEifQ.gva5d-xA6tC-y191P8wRPA")
				df = data.copy()
				fig = px.scatter_mapbox(data_frame=data, lat="lat", lon="long", zoom=15, hover_data=[
				                        "outgoing_site_id", "Traffic", "Call Dropped"], color="outgoing_site_id", size_max=100)
				st.plotly_chart(fig)
			except Exception as e:
				print(e)
		elif new == "Lineplot":
			col1.write('Please select the features:')
			siteid = col1.radio('Select Site ID', data['outgoing_site_id'].unique())
			df = data.loc[data['outgoing_site_id'] == siteid].reset_index()
			try:
				x_values = col1.selectbox('X-axis', options=numeric_columns)
				y_values = col1.selectbox('Y-axis', options=numeric_columns)
				plot = px.line(data_frame=df, x=x_values, y=y_values)
				with col2:
					st.write('Line plot for the Site ID:', siteid)
					st.plotly_chart(plot)
			except Exception as e:
				print(e)
		else:
			col1.write('Please select the features:')
			siteid = col1.radio('Select Site ID', data['outgoing_site_id'].unique())
			df = data.loc[data['outgoing_site_id'] == siteid].reset_index()
			try:
				x_values = col1.selectbox('X-axis', options=numeric_columns)
				y_values = col1.selectbox('Y-axis', options=numeric_columns)
				plot = px.pie(data_frame=data, names=x_values, values=y_values)
				with col2:
					st.write('Pie Chart for the Site ID:', siteid)
					st.plotly_chart(plot)
			except Exception as e:
				print(e)


#Model_training
else:
	col1, col2 = st.columns(2)
	with col1:
		st.header("Forecast")
		try:
			px.set_mapbox_access_token(
			    "pk.eyJ1IjoidmltYWwxMjM0IiwiYSI6ImNsMXN2dGlmMDI3cjgzY28yaXNxZWR3ZnEifQ.gva5d-xA6tC-y191P8wRPA")
			df = data.copy()
			fig = px.scatter_mapbox(data_frame=data, lat="lat", lon="long", zoom=15, hover_data=[
			                        "outgoing_site_id", "Traffic", "Call Dropped"], color="outgoing_site_id", size_max=100)
			st.plotly_chart(fig)
		except Exception as e:
			print(e)
		siteid = col1.radio('Select Site ID', data['outgoing_site_id'].unique())
	with col2:
		st.header("Forecast of Site ID")
		fdf = data.copy()
		#st.write(fdf)
		x_train, x_test, y_train, y_test, y_train_pred,prediction1= build_model(fdf)
		#st.write(x_test)
		x_test['Prediction'] = prediction1
		x_test = x_test.reset_index()
		x_test['date']= pd.to_datetime(x_test['Datetime'])
		x_test = x_test.set_index('date')
		x_train['Call Dropped']= y_train['Call Dropped']
		new = pd.concat([x_test, x_train])
		new = new.rename(columns={'Call Dropped':'Original_Value'})
		new = new.drop(columns = ['index'])
		a=new[new['outgoing_site_id']==siteid].astype(str)
		st.write(a)
		fig, ax = plt.subplots(figsize=(20, 6))
		ax.plot(a['Original_Value'],marker='.', linestyle='-', linewidth=0.5, label='Original')
		ax.plot(a['Prediction'],marker='o', linestyle='-', linewidth=0.5, label= 'Prediction')
		ax.legend()
		st.plotly_chart(fig)

