import streamlit as st
import pandas as pd
import os
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from regression_model import run_regression_model
from classification_model import run_classification_model
from time_series_model import run_time_series_model
import joblib  # Importing joblib for model loading

# Add custom CSS for a black theme

with st.sidebar:
    st.image("process.jpg")
    st.title("Auto StreamML")
    main_choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download", "Test Regression", "Test Classification", "Test Time_Series"])
    st.info("This application allows you to build an automated ML pipeline using Streamlit")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if main_choice == "Upload":
    st.title("Upload your dataset for modeling!")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        with st.expander("Data preview"):
            st.dataframe(df)

if main_choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = ProfileReport(df)
    st_profile_report(profile_report)

if main_choice == "ML":
    ml_sub_choice = st.radio("ML Task", ['Regression', 'Classification', 'Time_Series'])
    st.title('Machine Learning go')

    if ml_sub_choice == "Regression":
        run_regression_model(df)

    elif ml_sub_choice == 'Classification':
        run_classification_model(df)

    elif ml_sub_choice == 'Time_Series':
        run_time_series_model(df)

if main_choice == "Download":
    with open("best_model.pkl", "rb") as f:
        st.download_button("Download the file", f, "trained_model.pkl")

# Test Regression Model
if main_choice == "Test Regression":
    st.title("Test Regression Model")

    # Select target variable and input features
    target_variable = st.selectbox("Select the target variable", df.columns)
    input_features = st.multiselect("Select the input features", df.columns)

    if st.button("Test Regression Model"):
        # Load the saved regression model
        with open("Regression_best_model.pkl", "rb") as model_file:
            regression_model = joblib.load(model_file)

        # Example: Predict using the loaded regression model
        # You may need to modify this based on your model and input features
        X_test = df[input_features]
        y_true = df[target_variable]
        y_pred = regression_model.predict(X_test)

        # Display results
        result_df = pd.DataFrame({'True Value': y_true, 'Predicted Value': y_pred})
        st.dataframe(result_df)
        st.info(f"R-squared Score: {r2_score(y_true, y_pred)}")
        st.info(f"Mean Squared Error: {mean_squared_error(y_true, y_pred)}")

# Test Classification Model
if main_choice == "Test Classification":
    st.title("Test Classification Model")

    # Select target variable and input features
    target_variable = st.selectbox("Select the target variable", df.columns)
    input_features = st.multiselect("Select the input features", df.columns)

    if st.button("Test Classification Model"):
        # Load the saved classification model
        with open("Classification_best_model.pkl", "rb") as model_file:
            classification_model = joblib.load(model_file)

        # Example: Predict using the loaded classification model
        # You may need to modify this based on your model and input features
        X_test = df[input_features]
        y_true = df[target_variable]
        y_pred = classification_model.predict(X_test)

        # Display results
        result_df = pd.DataFrame({'True Value': y_true, 'Predicted Value': y_pred})
        st.dataframe(result_df)
        st.info(f"Accuracy Score: {accuracy_score(y_true, y_pred)}")

# Test Time Series Model
if main_choice == "Test Time_Series":
    st.title("Test Time Series Model")

    # Select target variable and input features
    target_variable = st.selectbox("Select the target variable", df.columns)
    input_features = st.multiselect("Select the input features", df.columns)

    if st.button("Test Time Series Model"):
        # Load the saved time series model
        with open("time_series_model.pkl", "rb") as model_file:
            time_series_model = joblib.load(model_file)

        # Example: Predict using the loaded time series model
        # You may need to modify this based on your model and input features
        X_test = df[input_features]
        y_true = df[target_variable]
        y_pred = time_series_model.predict(X_test)

        # Display results
        result_df = pd.DataFrame({'True Value': y_true, 'Predicted Value': y_pred})
        st.dataframe(result_df)
        # Add any specific metrics for time series if needed
