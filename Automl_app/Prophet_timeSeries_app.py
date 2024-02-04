import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go  # Import go module from plotly.graph_objects
import plotly.express as px

# Load sample data (replace with your own dataset)
@st.cache
def load_data():
    # Load your dataset here
    # For example, assuming you have a CSV file named 'your_data.csv'
    df = pd.read_csv('weather_data.csv')
    # df = df.rename(columns={'Date': 'ds', 'temp': 'y'})  # Adjust column names as needed
    df.columns=['ds', 'y']  # Adjust column names as needed
    return df

data = load_data()

# Sidebar for user input
st.sidebar.header('Prophet Time Series Forecasting')

# Select target variable
target_variable = st.sidebar.selectbox('Select Target Variable', data.columns)

# Sidebar section for forecasting parameters
st.sidebar.header('Prophet Parameters')

# Select forecast horizon
forecast_horizon = st.sidebar.slider('Select Forecast Horizon', min_value=1, max_value=365, value=30)

# Select changepoint range
changepoint_range = st.sidebar.slider('Select Changepoint Range', min_value=0.01, max_value=1.0, value=0.05, step=0.01)

# Create a Prophet instance with selected parameters
prophet_model = Prophet(changepoint_range=changepoint_range)

# Sidebar section for prediction
st.sidebar.header('Prediction')

# Radio button for selecting functionality
selected_functionality = st.sidebar.radio("Select Functionality", ["Run Forecast", "Prediction"])

if selected_functionality == "Run Forecast":
    # Perform Time Series Forecasting with Prophet
    if st.sidebar.button('Run Forecast'):
        # Fit the model
        prophet_model.fit(data)

        # Create future dates
        future = prophet_model.make_future_dataframe(periods=forecast_horizon)

        # Make predictions
        forecast = prophet_model.predict(future)

        # Separate historical data and forecasted values
        historical_data = data[['ds', 'y']]
        forecast_values = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        # Display forecast plot
        st.write('## Time Series Forecast')
        fig = px.line()
        fig.add_trace(go.Scatter(x=historical_data['ds'], y=historical_data['y'], mode='lines', name='Historical Data'))
        fig.add_trace(go.Scatter(x=forecast_values['ds'], y=forecast_values['yhat'], mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(x=forecast_values['ds'], y=forecast_values['yhat_lower'], mode='lines', name='Lower Bound'))
        fig.add_trace(go.Scatter(x=forecast_values['ds'], y=forecast_values['yhat_upper'], mode='lines', name='Upper Bound'))
        st.plotly_chart(fig)

elif selected_functionality == "Prediction":
    # File uploader for new dataset
    new_data_file = st.sidebar.file_uploader("Upload New Dataset for Prediction", type="csv")

    if new_data_file is not None:
        new_data = pd.read_csv(new_data_file)
        new_data = new_data.rename(columns={'Date': 'ds'})  # Adjust column names as needed

        # Fit the model with the historical data
        prophet_model.fit(data)

        # Make predictions on the new dataset
        future_data = prophet_model.make_future_dataframe(periods=len(new_data))
        forecast_data = prophet_model.predict(future_data)

        # Separate historical data and forecasted values for new data
        historical_data_new = new_data[['ds', 'y']]
        forecast_values_new = forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        # Display forecast plot for new data
        st.write('## Forecast Plot for New Dataset')
        fig = px.line()
        fig.add_trace(go.Scatter(x=historical_data_new['ds'], y=historical_data_new['y'], mode='lines', name='Historical Data'))
        fig.add_trace(go.Scatter(x=forecast_values_new['ds'], y=forecast_values_new['yhat'], mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(x=forecast_values_new['ds'], y=forecast_values_new['yhat_lower'], mode='lines', name='Lower Bound'))
        fig.add_trace(go.Scatter(x=forecast_values_new['ds'], y=forecast_values_new['yhat_upper'], mode='lines', name='Upper Bound'))
        st.plotly_chart(fig)

# Display the data
st.write('## Historical Data and Forecast')
st.write(data)
