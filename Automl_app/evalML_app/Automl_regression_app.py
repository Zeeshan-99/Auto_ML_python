import streamlit as st
import pandas as pd
import evalml
from evalml import AutoMLSearch
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import shutil
import os

# Load sample data (replace with your own dataset)
@st.cache
def load_data():
    # Load your dataset here
    # For example, assuming you have a CSV file named 'your_data.csv'
    df = pd.read_csv('laptop_data.csv')
    df = df.drop('Unnamed: 0', axis=1)
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

# Sidebar section for prediction
st.sidebar.header('Prediction')

# Radio button for selecting functionality
selected_functionality = st.sidebar.radio("Select Functionality", ["Run AutoML", "Prediction"])

if selected_functionality == "Run AutoML":
    # Perform AutoML Regression
    if st.sidebar.button('Run AutoML Regression'):
        # Clear existing model directory
        model_path = 'mlflow_best_model'
        if os.path.exists(model_path):
            shutil.rmtree(model_path)

        # Create an empty directory
        os.makedirs(model_path)

        automl = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type='regression', objective='r2')
        automl.search()

        # Display model rankings
        st.write('## Model Rankings')
        rankings = automl.rankings
        st.write(rankings)

        # Get the best pipeline
        best_pipeline = automl.best_pipeline

        # Save the best model as a pickle file
        mlflow.sklearn.save_model(best_pipeline, model_path)
        st.success('Best model saved as mlflow_best_model')

        # Plot actual vs predicted values
        st.write('## Actual vs Predicted Values')
        df_results = pd.DataFrame({'Actual': y_test, 'Predicted': best_pipeline.predict(X_test)})
        st.line_chart(df_results)

elif selected_functionality == "Prediction":
    # File uploader for new dataset
    new_data_file = st.sidebar.file_uploader("Upload New Dataset for Prediction", type="csv")

    if new_data_file is not None:
        new_data = pd.read_csv(new_data_file)

        # Load the saved model
        loaded_model = mlflow.sklearn.load_model('mlflow_best_model')

        # Make predictions on the new dataset
        predictions = loaded_model.predict(new_data)

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
