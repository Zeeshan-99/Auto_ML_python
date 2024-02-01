import streamlit as st
import pandas as pd
from evalml import AutoMLSearch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import mlflow
import mlflow.sklearn
import pickle
from ydata_profiling import ProfileReport
import io

# Function to perform data profiling
def perform_data_profiling(data):
    profile = ProfileReport(data)
    return profile

# Function to upload data
def upload_data():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success("Data uploaded successfully!")
        return data
    return None

# Sidebar for user input
st.sidebar.header('AutoML Regression')

# Upload data
data = upload_data()

# Perform data profiling
if data is not None and st.sidebar.button('Perform Data Profiling'):
    profile = perform_data_profiling(data)
    st.write('## Data Profiling Results')
    st.write(profile)

# Select target variable
if data is not None:
    st.sidebar.header('Data Information')
    st.sidebar.write(f'Selected File: {uploaded_file.name}')
    target_variable = st.sidebar.selectbox('Select Target Variable', data.columns)

    # Select features
    feature_columns = st.sidebar.multiselect('Select Features', data.columns)

    # Split the data into training and testing sets
    X = data[feature_columns]
    y = data[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform AutoML Regression
    if st.sidebar.button('Run AutoML Regression'):
        automl = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type='regression', objective='R2', max_iterations=10)
        automl.search()

        # Display model rankings
        st.write('## Model Rankings')
        rankings = automl.rankings.sort_values(by='R2', ascending=False)
        st.write(rankings)

        # Get the best pipeline
        best_pipeline = automl.best_pipeline

        # Make predictions on the test set
        y_pred = best_pipeline.predict(X_test)

        # Evaluate the model
        r2 = r2_score(y_test, y_pred)
        st.write(f'R2 Score for Best Model: {r2}')

        # Save the best model as a pickle file
        mlflow.sklearn.save_model(best_pipeline, 'mlflow_best_model')
        st.success('Best model saved as mlflow_best_model')

        # Plot actual vs predicted values
        st.write('## Actual vs Predicted Values')
        df_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        st.line_chart(df_results)

        # Display all R2 scores for all models
        st.write('## All R2 Scores for Models (Descending Order)')
        st.table(rankings[['Pipelines', 'R2']].sort_values(by='R2', ascending=False))
    
# Display the data
st.sidebar.header('Upload Data')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    st.sidebar.success(f"Selected File: {uploaded_file.name}")

st.sidebar.header('Data Profiling')
if uploaded_file is not None and st.sidebar.button('Perform Data Profiling'):
    data_for_profiling = pd.read_csv(uploaded_file)
    profile = perform_data_profiling(data_for_profiling)
    st.sidebar.write('## Data Profiling Results')
    st.sidebar.write(profile)
