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

st.title("📩 Text Spam Classifier")

msg = st.text_area("Enter message")

if st.button("Predict"):
    if msg.strip() == "":
        st.warning("Enter a message first")
    else:
        cleaned = clean_text(msg)
        X = vectorizer.transform([cleaned])

        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]

        ham_prob = probs[0] * 100
        spam_prob = probs[1] * 100

        st.success(f"Prediction: {pred.upper()}")

        st.subheader("Confidence")

        st.write(f"Ham: {ham_prob:.2f}%")
        st.progress(int(ham_prob))

        st.write(f"Spam: {spam_prob:.2f}%")
        st.progress(int(spam_prob))
