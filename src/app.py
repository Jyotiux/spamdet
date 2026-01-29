import streamlit as st
import joblib
from pathlib import Path
from preprocessing import clean_text

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


@st.cache_resource
def load_models():
    vect = joblib.load(MODEL_DIR / "vectorizer.joblib")
    model = joblib.load(MODEL_DIR / "logreg_model.joblib")
    return vect, model

vectorizer, model = load_models()

st.title("📧 SMS Spam Detector")

msg = st.text_area("Enter message")

if st.button("Predict"):
    if msg.strip() == "":
        st.warning("Enter a message first")
    else:
        cleaned = clean_text(msg)
        X = vectorizer.transform([cleaned])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X).max()

        st.success(f"Prediction: {pred}")
        st.write(f"Confidence: {prob:.2f}")
