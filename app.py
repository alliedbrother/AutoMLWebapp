from operator import index
import streamlit as st
from PIL import Image 
import plotly.express as px
import ydata_profiling
import pandas as pd
import matplotlib
from streamlit_pandas_profiling import st_profile_report
import os
import pycaret.classification as pyc 
import pycaret.regression as pyr


st.set_page_config(layout="wide")
image = Image.open('aml.png')

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv",index_col=None)

with st.sidebar:
    st.image(image,width = 75) 
    st.title("Auto ML - Just Build")
    choice = st.radio("Navigation",["Upload","Profiling","Auto ML","Download","Custom ML"])
    st.info("This application allows you to build an Auto ML model | ML is magic")

if choice == "Upload":
    st.title("Upload your data for modelling")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df = pd.read_csv(file,index_col=None)
        st.dataframe(df[0:25])
        df.to_csv("sourcedata.csv",index=None)
    if os.path.exists("sourcedata.csv"):
        df = pd.read_csv("sourcedata.csv",index_col=None)
        st.dataframe(df[0:25])


if choice == "Profiling":
    st.title("Explanatory Data Analysis")
    prof_report = df.profile_report()
    st_profile_report(prof_report)

if choice == "Auto ML":
    chosen_model_type = st.selectbox('Dataset for Regression or Classification', ["Regression","Classification"])
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    
    if chosen_model_type == "Classification":
        if st.button('Give Parameters'):
            pyc.setup(df, target=chosen_target) 
            setup_df = pyc.pull() 
            #st.dataframe(setup_df,use_container_width=True)
        if st.button('Build Model'):
            with st.spinner("Building your Model"):
                best_model = pyc.compare_models()
                compare_df = pyc.pull()
                st.dataframe(compare_df,use_container_width=True)
                best_model
                pyc.save_model(best_model, 'best_model')

    if chosen_model_type == "Regression":
        if st.button('Give Parameters'):
            pyr.setup(df, target=chosen_target)
            setup_df = pyr.pull()
            st.dataframe(setup_df,use_container_width=True)
        if st.button('Build Model'):
            with st.spinner("Building your Model"):
                best_model = pyr.compare_models()
                compare_df = pyr.pull()
                st.dataframe(compare_df,use_container_width=True)
                best_model
                pyr.save_model(best_model, 'best_model')

if choice == "Download":
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")

if choice == "Custom ML":
    st.write("More features coming soon")

