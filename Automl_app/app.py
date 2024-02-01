import streamlit as st
import pandas as pd
import os
# import profiling capabilites
# import pandas_profiling
from streamlit_pandas_profiling import st_profile_report


with st.sidebar:
    # st.image("https://incubator.ucf.edu/wp-content/uploads/2023/…d-domination-generative-ai-scaled-1-1500x1000.jpg")
    st.image("process.jpg")
    st.title("Auto StreamML")
    choice= st.radio("Navigation", ["Upload","Profiling","ML","Download"])
    st.info("This application allows you to build an automated ML pipeline using Streamlit")


if os.path.exists("sourcedata.csv"):
    df= pd.read_csv("sourcedata.csv", index_col=None)
    
if choice == "Upload":
    st.title("Upload Your dataset for modeling!")
    file=st.file_uploader("Upload Your Dataset Here")
    if file:
        df= pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)
        
if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report =df.profile_report()
    st_profile_report(profile_report)
    
if choice =="ML":
    pass
if choice == "Download":
    pass