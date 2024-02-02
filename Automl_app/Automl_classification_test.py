import streamlit as st
import pandas as pd
import evalml
from evalml import AutoMLSearch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import mlflow
import mlflow.sklearn
import shutil
import os
import plotly.express as px
import plotly.figure_factory as ff
from ydata_profiling import ProfileReport

# Load sample data (replace with your own dataset)
@st.cache
def load_data():
    # Load your dataset here
    # For example, assuming you have a CSV file named 'your_data.csv'
    df = pd.read_csv('iris.csv')
    return df

data = load_data()

# Display the ProfileReport immediately after loading data
st.write('## Data Profile Report')
profile_report = ProfileReport(data)
st.write(profile_report.to_html(), unsafe_allow_html=True)  # Assuming a to_html() method exists
# st.write(profile_report)

# Sidebar for user input
st.sidebar.header('AutoML Classification')

# Select target variable
target_variable = st.sidebar.selectbox('Select Target Variable', data.columns)

# Select features
feature_columns = st.sidebar.multiselect('Select Features', data.columns)

# Split the data into training and testing sets
X = data[feature_columns]
y = data[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar section for prediction
st.sidebar.header('Prediction')

# Radio button for selecting functionality
selected_functionality = st.sidebar.radio("Select Functionality", ["Run AutoML", "Prediction"])

if selected_functionality == "Run AutoML":
    # Perform AutoML Classification
    if st.sidebar.button('Run AutoML Classification'):
        # Clear existing model directory
        model_path = 'mlflow_best_model'
        if os.path.exists(model_path):
            shutil.rmtree(model_path)

        # Create an empty directory
        os.makedirs(model_path)

        automl = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type='multiclass', objective='log loss multiclass', max_iterations=20)
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

        # Display evaluation metrics
        y_pred = best_pipeline.predict(X_test)
        y_pred_proba = best_pipeline.predict_proba(X_test)

        st.write('## Evaluation Metrics')
        st.write(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
        st.write(f'Classification Report:\n{classification_report(y_test, y_pred)}')

        # Calculate and display AUC score for each class
        st.write('## AUC Score for Each Class')
        class_auc_scores = []
        for i in range(len(y_test.unique())):
            class_auc_scores.append({
                'Class': i,
                'AUC Score': roc_auc_score(y_test == i, y_pred_proba[:, i])
            })
        auc_df = pd.DataFrame(class_auc_scores)
        st.write(auc_df)

        # Plot confusion matrix as an image
        st.write('## Confusion Matrix')
        confusion_matrix_fig = px.imshow(confusion_matrix(y_test, y_pred),
                                        labels=dict(x="Predicted", y="True", color="Count"))
        st.plotly_chart(confusion_matrix_fig)

        # Plot ROC curves
        st.write('## ROC Curves for Each Class')
        for i in range(len(y_test.unique())):
            fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            roc_curve_fig = px.area(x=fpr, y=tpr, title=f'ROC Curve - Class {i}, AUC={roc_auc:.2f}',
                                    labels=dict(x='False Positive Rate', y='True Positive Rate'))
            st.plotly_chart(roc_curve_fig)


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

# Display the data
st.write('## Dataset')
st.write(data)
