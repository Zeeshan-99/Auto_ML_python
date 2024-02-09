import streamlit as st
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import plotly.express as px

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
st.sidebar.header('Forecasting Parameters')

# Select forecast horizon
forecast_horizon = st.sidebar.slider('Select Forecast Horizon', min_value=1, max_value=365, value=30)

# Select model
selected_model = st.sidebar.selectbox('Select Model', ['Prophet', 'SARIMA'])

# Create a model instance based on user selection
if selected_model == 'Prophet':
    model = Prophet()
elif selected_model == 'SARIMA':
    model = SARIMAX(data['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

# Sidebar section for prediction
st.sidebar.header('Prediction')

# Radio button for selecting functionality
selected_functionality = st.sidebar.radio("Select Functionality", ["Run Forecast", "Prediction"])

if selected_functionality == "Run Forecast":
    # Perform Time Series Forecasting
    if st.sidebar.button('Run Forecast'):
        if selected_model == 'Prophet':
            # Fit the model for Prophet
            model_fit = model.fit(data)
            
            # Create future dates for Prophet
            future_dates = pd.date_range(start=data['ds'].max(), periods=forecast_horizon, freq='D')
            future = pd.DataFrame({'ds': future_dates})

            # Make predictions
            forecast = model_fit.predict(future)

            # Evaluate model performance for Prophet
            metrics = {
                'MAE': mean_absolute_error(data['y'], forecast['yhat'][:-forecast_horizon]),
                'MSE': mean_squared_error(data['y'], forecast['yhat'][:-forecast_horizon]),
                'RMSE': mean_squared_error(data['y'], forecast['yhat'][:-forecast_horizon], squared=False)
            }
            
        elif selected_model == 'SARIMA':
            # Fit the model for SARIMA
            model_fit = model.fit()

            # Forecast for SARIMA
            forecast = model_fit.get_forecast(steps=forecast_horizon)

            # Evaluate model performance for SARIMA
            metrics = {
                'MAE': mean_absolute_error(data['y'][-forecast_horizon:], forecast.predicted_mean),
                'MSE': mean_squared_error(data['y'][-forecast_horizon:], forecast.predicted_mean),
                'RMSE': mean_squared_error(data['y'][-forecast_horizon:], forecast.predicted_mean, squared=False)
            }

        # Display evaluation metrics
        st.write('## Model Evaluation Metrics')
        st.write(metrics)

        # Separate historical data and forecasted values
        historical_data = data[['ds', 'y']]
        forecast_values = forecast[['ds', 'yhat' if selected_model == 'Prophet' else 'predicted_mean']]

        # Display forecast plot
        st.write('## Time Series Forecast')
        fig = px.line()
        fig.add_trace(go.Scatter(x=historical_data['ds'], y=historical_data['y'], mode='lines', name='Historical Data'))
        fig.add_trace(go.Scatter(x=forecast_values['ds'], y=forecast_values['yhat' if selected_model == 'Prophet' else 'predicted_mean'], mode='lines', name='Forecast'))

        if selected_model == 'Prophet':
            # Plot 95% confidence interval for Prophet
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
            model_fit = model.fit(data)
            
            # Make predictions on the new dataset
            future_data = model.make_future_dataframe(periods=len(new_data))
            forecast_data = model_fit.predict(future_data)

        elif selected_model == 'SARIMA':
            model_fit = model.fit()

            # Forecast for SARIMA
            forecast_data = model_fit.get_forecast(steps=len(new_data))

        # Evaluate model performance for new data
        if selected_model == 'Prophet':
            metrics_new_data = {
                'MAE': mean_absolute_error(new_data['y'], forecast_data['yhat']),
                'MSE': mean_squared_error(new_data['y'], forecast_data['yhat']),
                'RMSE': mean_squared_error(new_data['y'], forecast_data['yhat'], squared=False)
            }

        elif selected_model == 'SARIMA':
            metrics_new_data = {
                'MAE': mean_absolute_error(new_data['y'], forecast_data.predicted_mean),
                'MSE': mean_squared_error(new_data['y'], forecast_data.predicted_mean),
                'RMSE': mean_squared_error(new_data['y'], forecast_data.predicted_mean, squared=False)
            }

        # Display evaluation metrics for new data
        st.write('## Model Evaluation Metrics for New Dataset')
        st.write(metrics_new_data)

        # Separate historical data and forecasted values for new data
        historical_data_new = new_data[['ds', 'y']]
        forecast_values_new = forecast_data[['ds', 'yhat' if selected_model == 'Prophet' else 'predicted_mean']]

        # Display forecast plot for new data
        st.write('## Forecast Plot for New Dataset')
        fig = px.line()
        fig.add_trace(go.Scatter(x=historical_data_new['ds'], y=historical_data_new['y'], mode='lines', name='Historical Data'))
        fig.add_trace(go.Scatter(x=forecast_values_new['ds'], y=forecast_values_new['yhat' if selected_model == 'Prophet' else 'predicted_mean'], mode='lines', name='Forecast'))

        if selected_model == 'Prophet':
            # Plot 95% confidence interval for Prophet
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
