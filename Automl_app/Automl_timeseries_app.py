import streamlit as st
import pandas as pd
from pycaret.regression import *
from pycaret.utils import check_metric
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load sample data (replace with your own dataset)
@st.cache_resource
def load_data():
    # Load your dataset here
    # For example, assuming you have a CSV file named 'your_data.csv'
    df = pd.read_csv('time_data.csv')
    # df = df.drop('Unnamed: 0', axis=1)
    return df

data = load_data()

# Sidebar for user input
st.sidebar.header('Time Series Forecasting with PyCaret')

# Select target variable
target_variable = st.sidebar.selectbox('Select Target Variable', data.columns)

# Select features
feature_columns = st.sidebar.multiselect('Select Features', data.columns)

# Sidebar section for prediction
st.sidebar.header('Prediction')

# Radio button for selecting functionality
selected_functionality = st.sidebar.radio("Select Functionality", ["Run PyCaret", "Prediction"])

if selected_functionality == "Run PyCaret":
    # Setup PyCaret environment
    exp = setup(
        data=data,
        target=target_variable,
        fold_strategy="timeseries",  # Use timeseries split
        numeric_features=feature_columns,
        preprocess=True,
        session_id=123  # Optional for reproducibility
    )

    # Create a time series experiment
    best_model = create_model('ts')

    # Display model results
    st.write('## Model Results')
    st.write(best_model)

    # Plot actual vs predicted values
    st.write('## Actual vs Predicted Values')
    plot_model(best_model)
    st.pyplot()

elif selected_functionality == "Prediction":
    # File uploader for new dataset
    new_data_file = st.sidebar.file_uploader("Upload New Dataset for Prediction", type="csv")

    if new_data_file is not None:
        new_data = pd.read_csv(new_data_file)

        # Make predictions on the new dataset
        predictions = predict_model(best_model, data=new_data)

        # Display predictions
        st.write('## Predictions on New Dataset')
        st.write(predictions)

        # Generate plots based on predictions (modify as needed)
        # For example, you can create a histogram of predicted values
        st.write('## Histogram of Predicted Values')
        plt.hist(predictions, bins=20, color='skyblue', edgecolor='black')
        st.pyplot()

# Display the data
st.write('## Dataset')
st.write(data)
