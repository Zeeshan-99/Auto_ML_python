import streamlit as st
import pandas as pd
import os
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from regression_model import run_regression_model
from classification_model import run_classification_model
from time_series_model import run_time_series_model

# Add custom CSS for a black theme

with st.sidebar:
    st.image("process.jpg")
    st.title("Auto StreamML")
    main_choice= st.radio("Navigation", ["Upload","Profiling","ML","Download"])
    st.info("This application allows you to build an automated ML pipeline using Streamlit")

if os.path.exists("sourcedata.csv"):
    df= pd.read_csv("sourcedata.csv", index_col=None)

if main_choice == "Upload":
    st.title("Upload your dataset for modeling!")
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
                
            
if main_choice == "ML":
    ml_sub_choice = st.radio("ML Task",['Regression','Classification','Time_Series'])
    st.title('Machine Learning go')

    if ml_sub_choice == "Regression":
        run_regression_model(df)

    elif ml_sub_choice == 'Classification':
        run_classification_model(df)

    elif ml_sub_choice == 'Time_Series':
        run_time_series_model(df)  

if main_choice == "Download":
    with open("best_model.pkl","rb") as f:
        st.download_button("Download the file",f,"trained_model.pkl")
    