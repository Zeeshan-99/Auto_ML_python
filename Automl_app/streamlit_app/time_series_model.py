from prophet import Prophet
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_time_series_model(df):
    
    st.info("You selected Time Series task.")
    df.columns=['ds','y']
    st.dataframe(df)
    
    if st.button("Predict"):
        # Model building
        # Sidebar for user input
        st.sidebar.header('Prophet Time Series Forecasting')

        # Select target variable
        target_variable = st.sidebar.selectbox('Select Target Variable', df.columns)

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
                prophet_model.fit(df)

                # Create future dates
                future = prophet_model.make_future_dataframe(periods=forecast_horizon)

                # Make predictions
                forecast = prophet_model.predict(future)

                # Evaluate model performance
                metrics = {
                    'MAE': mean_absolute_error(df['y'], forecast['yhat'][:-forecast_horizon]),
                    'MSE': mean_squared_error(df['y'], forecast['yhat'][:-forecast_horizon]),
                    'RMSE': mean_squared_error(df['y'], forecast['yhat'][:-forecast_horizon], squared=False)
                }

                # Display evaluation metrics
                st.write('## Model Evaluation Metrics')
                st.write(metrics)

                # Separate historical data and forecasted values
                historical_data = df[['ds', 'y']]
                forecast_values = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

                # Display forecast plot
                st.write('## Time Series Forecast')
                fig = go.Figure()

                # Plot historical data
                fig.add_trace(go.Scatter(x=historical_data['ds'], y=historical_data['y'], mode='lines', name='Historical Data'))

                # Plot forecasted values after historical data
                fig.add_trace(go.Scatter(x=forecast_values['ds'], y=forecast_values['yhat'], mode='lines', name='Forecast'))

                # Plot 95% confidence interval
                fig.add_trace(go.Scatter(x=forecast_values['ds'], y=forecast_values['yhat_upper'], mode='lines', line=dict(color='rgba(0,100,80,0.2)'), name='Upper Bound'))
                fig.add_trace(go.Scatter(x=forecast_values['ds'], y=forecast_values['yhat_lower'], mode='lines', line=dict(color='rgba(0,100,80,0.2)'), name='Lower Bound'))

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
                prophet_model.fit(data)

                # Make predictions on the new dataset
                future_data = prophet_model.make_future_dataframe(periods=len(new_data))
                forecast_data = prophet_model.predict(future_data)

                # Separate historical data and forecasted values for new data
                historical_data_new = new_data[['ds', 'y']]
                forecast_values_new = forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

                # Display forecast plot for new data
                st.write('## Forecast Plot for New Dataset')
                fig = go.Figure()

                # Plot historical data
                fig.add_trace(go.Scatter(x=historical_data_new['ds'], y=historical_data_new['y'], mode='lines', name='Historical Data'))

                # Plot forecasted values
                fig.add_trace(go.Scatter(x=forecast_values_new['ds'], y=forecast_values_new['yhat'], mode='lines', name='Forecast'))
                fig.add_trace(go.Scatter(x=forecast_values_new['ds'], y=forecast_values_new['yhat_lower'], mode='lines', name='Lower Bound'))
                fig.add_trace(go.Scatter(x=forecast_values_new['ds'], y=forecast_values_new['yhat_upper'], mode='lines', name='Upper Bound'))

                # Set layout for a professional appearance
                fig.update_layout(
                    title='Forecast Plot for New Dataset',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    template='plotly_dark',  # Choose a template that suits your style
                    showlegend=True,
                )

                st.plotly_chart(fig)

        # Display the data
        st.write('## Historical Data and Forecast')
        st.write(df)
