


import numpy as np
import pickle
import pandas as pd
import streamlit as st

pickle_in = open('rf_clf.pkl','rb')
classifier=pickle.load(pickle_in)

def welcome():
    return "Welcome All"

def predict_note_authentication(Undergrad,Marital_Status,Taxable_Income,City_Population,Work_Experience):
    prediction=classifier.predict([[Undergrad,Marital_Status,Taxable_Income,City_Population,Work_Experience]])
    print(prediction)
    return prediction

def main():
    st.title("fraud_data")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">fraud_data ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Undergrad = st.text_input("Undergrad","Type Here")
    Marital_Status = st.text_input("Marital_Status","Type Here")
    Taxable_Income = st.text_input("Taxable_Income","Type Here")
    City_Population = st.text_input("City_Population","Type Here")
    Work_Experience = st.text_input("Work_Experience","Type Here")
    result=""
    
    if st.button("Predict"):
        result=predict_note_authentication(Undergrad,Marital_Status,Taxable_Income,City_Population,Work_Experience)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    
    

    #Date = st.text_input("Date","Type Here")
    #Machine_ID = st.text_input("Machine_ID","Type Here")