import streamlit as st
import pandas as pd
import os
import duckdb
import plotly.express as px
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model
from pycaret.regression import *
from pycaret.time_series import *
from prophet import Prophet

with st.sidebar:
    # st.image("https://incubator.ucf.edu/wp-content/uploads/2023/…d-domination-generative-ai-scaled-1-1500x1000.jpg")
    st.image("process.jpg")
    st.title("Auto StreamML")
    main_choice= st.radio("Navigation", ["Upload","Profiling","ML","Download"])
    st.info("This application allows you to build an automated ML pipeline using Streamlit")

if os.path.exists("sourcedata.csv"):
    df= pd.read_csv("sourcedata.csv", index_col=None)

if main_choice == "Upload":
    st.title("Upload Your dataset for modeling!")
    file=st.file_uploader("Upload Your Dataset Here")
    if file:
        df= pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        with st.expander("Data preview"):
            # st.dataframe(df, column_config={"Year": st.column_config.NumberColumn(format="%d")})
            st.dataframe(df)
            
if main_choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = ProfileReport(df)
    st_profile_report(profile_report)
    
if main_choice =="ML":
    ml_sub_choice = st.radio("ML Task",['Regression','Classification','Time_Series'])
    st.title('ML go')
    if ml_sub_choice == "Regression":
        st.info("You selected Regression task.")
        target = st.selectbox("Select Your Target", df.columns)
        df.fillna(value = df.mode().iloc[0], inplace=True)
        if st.button("Train regression model"):
            setup(df, target = target)
            setup_df = pull()
            st.info("This is the ML Experiment settings")
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.info("This is the ML Model")
            st.dataframe(compare_df)
            best_model
            save_model(best_model, 'Regression_best_model')

    elif ml_sub_choice == 'Classification':
        # Code for classification tasks
        st.info("You selected Classification task.")
        target = st.selectbox("Select Your Target", df.columns)
        df.fillna(value = df.mode().iloc[0], inplace=True)
        if st.button("Train classification model"):
            setup(df, target = target)
            setup_df = pull()
            st.info("This is the ML Experiment settings")
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.info("This is the ML Model")
            st.dataframe(compare_df)
            best_model
            save_model(best_model, 'Classification_best_model')

    elif ml_sub_choice == 'Time_Series':
        # Code for time series tasks
        st.info("You selected Time Series task.")
        df.columns=['ds','y']
        st.dataframe(df)
        if st.button("Predict"):
            # Model building
            model = Prophet()
            model.fit(df)

            # Forecasting
            future = model.make_future_dataframe(periods=90)  # 1 year forecast
            forecast = model.predict(future)
            forecast =forecast.set_index('ds')

            # Display forecast plot
            st.write("Forecast Plot:")
            st.dataframe(forecast)
            st.line_chart(forecast[['trend','yhat']]) 
        
if main_choice == "Download":
    with open("best_model.pkl","rb") as f:
        st.download_button("Download the file",f,"trained_model.pkl")
    