import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport

# Install the ydata library if not already installed
try:
    import ydata_profiling
except ImportError:
    st.error("Please install the ydata library: pip install ydata_profiling")

# Load data (replace with your file path)
@st.cache
def load_data():
    return pd.read_csv("iris.csv")

# Create profile report
def create_profile_report(data):
    report = ProfileReport(data)
    return report

# Main app structure
st.title("Data Profiling App using ydata")

df= load_data()
report=create_profile_report(df)
st.write(report.to_html(), unsafe_allow_html=True)

