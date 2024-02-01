import streamlit as st
import pandas as pd
import evalml
from evalml import AutoMLSearch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load sample data (replace with your own dataset)
@st.cache
def load_data():
    # Load your dataset here
    # For example, assuming you have a CSV file named 'your_data.csv'
    df= pd.read_csv('laptop_data.csv')
    df=df.drop('Unnamed: 0', axis=1)
    return df

data = load_data()

# Sidebar for user input
st.sidebar.header('AutoML Regression')

# Select target variable
target_variable = st.sidebar.selectbox('Select Target Variable', data.columns)

# Select features
feature_columns = st.sidebar.multiselect('Select Features', data.columns)

# Split the data into training and testing sets
X = data[feature_columns]
y = data[target_variable]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = evalml.preprocessing.split_data(X, y, problem_type='regression')

# Perform AutoML Regression
if st.sidebar.button('Run AutoML Regression'):
    # automl = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type='regression', objective='r2', max_iterations=10)
    automl = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type='regression', objective='r2')
    automl.search()

    # Display model rankings
    st.write('## Model Rankings')
    rankings = automl.rankings
    st.write(rankings)

    # Get the best pipeline
    best_pipeline = automl.best_pipeline

    # Make predictions on the test set
    y_pred = best_pipeline.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    st.write(f'Mean Squared Error (MSE) for Best Model: {mse}')

    # Save the best model as a pickle file
    mlflow.sklearn.save_model(best_pipeline, 'mlflow_best_model')
    st.success('Best model saved as mlflow_best_model')

    # Plot actual vs predicted values
    st.write('## Actual vs Predicted Values')
    df_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    st.line_chart(df_results)

# Display the data
st.write('## Dataset')
st.write(data)
