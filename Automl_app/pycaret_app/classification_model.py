from pycaret.classification import *
import streamlit as st

def run_classification_model(df):
    st.info("You selected Classification task.")
    target = st.selectbox("Select Your Target", df.columns)
    df.fillna(value = df.mode().iloc[0], inplace=True)
    
    if st.button("Train classification model"):
        setup(df, target=target)
        setup_df = pull()
        st.info("This is the ML Experiment settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'Classification_best_model')
