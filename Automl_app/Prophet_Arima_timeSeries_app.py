import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Load sample data (replace with your own dataset)
@st.cache
def load_data():
    # Load your dataset here
    # For example, assuming you have a CSV file named 'your_data.csv'
    df = pd.read_csv('suzlon.csv')
    df = df[['Date', 'Close']]
    df.columns = ['ds', 'y']  # Adjust column names as needed
    return df

data = load_data()

# Sidebar for user input
st.sidebar.header('Time Series Forecasting')

# Select target variable
target_variable = st.sidebar.selectbox('Select Target Variable', data.columns)

# Sidebar section for forecasting parameters
st.sidebar.header('Model Parameters')

# Select forecast horizon
forecast_horizon = st.sidebar.slider('Select Forecast Horizon', min_value=1, max_value=365, value=30)

# Select model
selected_model = st.sidebar.radio("Select Model", ["Prophet", "SARIMA"])

# Create a model instance with selected parameters
if selected_model == 'Prophet':
    model = Prophet()
elif selected_model == 'SARIMA':
    model = SARIMAX(data['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

# Sidebar section for prediction
st.sidebar.header('Functionality')

# Radio button for selecting functionality
selected_functionality = st.sidebar.radio("Select Functionality", ["Run Forecast", "Prediction"])

if selected_functionality == "Run Forecast":
    # Perform Time Series Forecasting
    if st.sidebar.button('Run Forecast'):
        # Fit the model
        if selected_model == 'Prophet':
            model.fit(data)
        elif selected_model == 'SARIMA':
            # Fit the SARIMA model
            model_fit = model.fit(disp=False)

        # Create future dates
        future_dates = pd.date_range(start=data['ds'].max(), periods=forecast_horizon + 1, freq='D')[1:]
        future = pd.DataFrame({'ds': future_dates})

        # Make predictions
        if selected_model == 'Prophet':
            forecast = model.predict(future)
        elif selected_model == 'SARIMA':
            forecast = model_fit.get_forecast(steps=forecast_horizon)

        # Evaluate model performance
        if selected_model == 'Prophet':
            if not forecast.empty:
                metrics = {
                    'MAE': mean_absolute_error(data['y'], forecast['yhat'][:-forecast_horizon]),
                    'MSE': mean_squared_error(data['y'], forecast['yhat'][:-forecast_horizon]),
                    'RMSE': mean_squared_error(data['y'], forecast['yhat'][:-forecast_horizon], squared=False)
                }
            else:
                metrics = {}
        elif selected_model == 'SARIMA':
            # You can add relevant metrics for SARIMA here
            metrics = {}

        # Display evaluation metrics
        st.write('## Model Evaluation Metrics')
        st.write(metrics)

        # Separate historical data and forecasted values
        historical_data = data[['ds', 'y']]
        forecast_values = forecast[['ds', 'yhat' if selected_model == 'Prophet' else 'predicted_mean']]

        # Display forecast plot
        st.write('## Time Series Forecast')
        fig = go.Figure()

        # Plot historical data
        fig.add_trace(go.Scatter(x=historical_data['ds'], y=historical_data['y'], mode='lines', name='Historical Data'))

        # Plot forecasted values after historical data
        fig.add_trace(go.Scatter(x=forecast_values['ds'], y=forecast_values['yhat' if selected_model == 'Prophet' else 'predicted_mean'], mode='lines', name='Forecast'))

        if selected_model == 'Prophet':
            # Plot 95% confidence interval
            fig.add_trace(go.Scatter(x=forecast_values['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(color='rgba(0,100,80,0.2)'), name='Upper Bound'))
            fig.add_trace(go.Scatter(x=forecast_values['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(color='rgba(0,100,80,0.2)'), name='Lower Bound'))

        # Set layout for a professional appearance
        fig.update_layout(
            title='Time Series Forecast',
            xaxis_title='Date',
            yaxis_title='Value',
            template='plotly_dark',  # Choose a template that suits your style
            showlegend=True,
        )

        st.plotly_chart(fig)

elif selected_functionality == "Prediction":
    # File uploader for new dataset
    new_data_file = st.sidebar.file_uploader("Upload New Dataset for Prediction", type="csv")

    if new_data_file is not None:
        new_data = pd.read_csv(new_data_file)
        new_data = new_data.rename(columns={'Date': 'ds'})  # Adjust column names as needed

        # Fit the model with the historical data
        if selected_model == 'Prophet':
            model.fit(data)
        elif selected_model == 'SARIMA':
            # Fit the SARIMA model
            model_fit = model.fit(disp=False)

        # Make predictions on the new dataset
        future_data = new_data.copy()
        future_data['ds'] = pd.date_range(start=new_data['ds'].max(), periods=len(new_data), freq='D')

        if selected_model == 'Prophet':
            forecast_data = model.predict(future_data[['ds']])
        elif selected_model == 'SARIMA':
            forecast_data = model_fit.get_forecast(steps=len(new_data))

        # Evaluate model performance for new data
        if selected_model == 'Prophet':
            if not forecast_data.empty:
                metrics_new_data = {
                    'MAE': mean_absolute_error(new_data['y'], forecast_data['yhat']),
                    'MSE': mean_squared_error(new_data['y'], forecast_data['yhat']),
                    'RMSE': mean_squared_error(new_data['y'], forecast_data['yhat'], squared=False)
                }
            else:
                metrics_new_data = {}
        elif selected_model == 'SARIMA':
            # You can add relevant metrics for SARIMA here
            metrics_new_data = {}

        # Display evaluation metrics for new data
        st.write('## Model Evaluation Metrics for New Dataset')
        st.write(metrics_new_data)

        # Separate historical data and forecasted values for new data
        historical_data_new = new_data[['ds', 'y']]
        forecast_values_new = forecast_data[['ds', 'yhat' if selected_model == 'Prophet' else 'predicted_mean']]

        # Display forecast plot for new data
        st.write('## Forecast Plot for New Dataset')
        fig = go.Figure()

        # Plot historical data for new data
        fig.add_trace(go.Scatter(x=historical_data_new['ds'], y=historical_data_new['y'], mode='lines', name='Historical Data'))

        # Plot forecasted values for new data
        fig.add_trace(go.Scatter(x=forecast_values_new['ds'], y=forecast_values_new['yhat' if selected_model == 'Prophet' else 'predicted_mean'], mode='lines', name='Forecast'))

        if selected_model == 'Prophet':
            # Plot 95% confidence interval for new data
            fig.add_trace(go.Scatter(x=forecast_values_new['ds'], y=forecast_data['yhat_upper'], mode='lines', line=dict(color='rgba(0,100,80,0.2)'), name='Upper Bound'))
            fig.add_trace(go.Scatter(x=forecast_values_new['ds'], y=forecast_data['yhat_lower'], mode='lines', line=dict(color='rgba(0,100,80,0.2)'), name='Lower Bound'))

        # Set layout for a professional appearance
        fig.update_layout(
            title='Time Series Forecast for New Dataset',
            xaxis_title='Date',
            yaxis_title='Value',
            template='plotly_dark',  # Choose a template that suits your style
            showlegend=True,
        )

        st.plotly_chart(fig)

# Display the data
st.write('## Historical Data')
st.write(data)
