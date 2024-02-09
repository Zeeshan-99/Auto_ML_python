import streamlit as st
import pandas as pd
from pycaret.regression import *

# Load sample data (replace with your own dataset)
@st.cache
def load_data():
    # Load your dataset here
    # For example, assuming you have a CSV file named 'your_data.csv'
    df = pd.read_csv('suzlon.csv')
    df = df[['Date', 'Close']]
    df.columns = ['ds', 'y']
    return df

data = load_data()

# Sidebar for user input
st.sidebar.header('Pycaret Time Series Forecasting')

# Select target variable
target_variable = st.sidebar.selectbox('Select Target Variable', data.columns)

# Sidebar section for forecasting parameters
st.sidebar.header('Pycaret Time Series Parameters')

# Select forecast horizon
forecast_horizon = st.sidebar.slider('Select Forecast Horizon', min_value=1, max_value=365, value=30)

# Sidebar section for prediction
st.sidebar.header('Prediction')

# Radio button for selecting functionality
selected_functionality = st.sidebar.radio("Select Functionality", ["Run Forecast", "Prediction"])

if selected_functionality == "Run Forecast":
    # Perform Time Series Forecasting with Pycaret
    if st.sidebar.button('Run Forecast'):
        # Prepare the data for Pycaret
        train_data = data.copy()
        
        # Initialize Pycaret setup
        exp = setup(data=train_data, target='y', session_id=123, silent=True, profile=False,
                    ignore_features=['ds'])  # Ignore 'ds' column as it represents the time variable

        # Create and compare models
        best_model = compare_models(fold=5, round=2, sort='MAE')

        # Tune the best model
        tuned_model = tune_model(best_model, fold=5, round=2, n_iter=10)

        # Forecast using the tuned model
        predictions = predict_model(tuned_model, data)

        # Evaluate model performance
        metrics = {
            'MAE': mean_absolute_error(predictions['y'], predictions['Label']),
            'MSE': mean_squared_error(predictions['y'], predictions['Label']),
            'RMSE': mean_squared_error(predictions['y'], predictions['Label'], squared=False)
        }

        # Display evaluation metrics
        st.write('## Model Evaluation Metrics')
        st.write(metrics)

        # Display forecast plot
        st.write('## Time Series Forecast')
        fig = plot_model(tuned_model, plot='residuals_interactive')

        st.pyplot(fig)

elif selected_functionality == "Prediction":
    # File uploader for new dataset
    new_data_file = st.sidebar.file_uploader("Upload New Dataset for Prediction", type="csv")

    if new_data_file is not None:
        new_data = pd.read_csv(new_data_file)
        new_data = new_data.rename(columns={'Date': 'ds'})  # Adjust column names as needed

        # Prepare the data for Pycaret
        train_data = data.copy()
        
        # Initialize Pycaret setup
        exp = setup(data=train_data, target='y', session_id=123, silent=True, profile=False,
                    ignore_features=['ds'])  # Ignore 'ds' column as it represents the time variable

        # Create and compare models
        best_model = compare_models(fold=5, round=2, sort='MAE')

        # Tune the best model
        tuned_model = tune_model(best_model, fold=5, round=2, n_iter=10)

        # Forecast on the new dataset using the tuned model
        forecast_data = predict_model(tuned_model, new_data)

        # Display forecast plot for new data
        st.write('## Forecast Plot for New Dataset')
        fig = plot_model(tuned_model, plot='forecast', data=forecast_data)

        st.pyplot(fig)

# Display the data
st.write('## Historical Data and Forecast')
st.write(data)
