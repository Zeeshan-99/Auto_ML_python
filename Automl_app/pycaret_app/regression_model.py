from pycaret.regression import *
from sklearn.metrics import r2_score, mean_squared_error
import streamlit as st

def run_regression_model(df):
    st.info("You selected Regression task.")
    target = st.selectbox("Select Your Target", df.columns)
    df.fillna(value = df.mode().iloc[0], inplace=True)
    
    if st.button("Train regression model"):
        setup(df, target=target)
        setup_df = pull()
        st.info("This is the ML Experiment settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML Model")
        
        # Evaluate the model
        evaluate_model(best_model)
        # Interpret the model
        interpret_model(best_model)
        
        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'Regression_best_model')
