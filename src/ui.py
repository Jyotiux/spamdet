import streamlit as st
import requests

st.title("SMS Spam Detector")

msg = st.text_area("Enter message")

if st.button("Predict"):
    r = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"message": msg}
    )
    st.write(r.json())
