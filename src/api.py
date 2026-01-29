"""FastAPI app exposing a /predict endpoint for SMS spam detection."""
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# ensure local imports work when running from project root
HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))

from preprocessing import clean_text


class PredictRequest(BaseModel):
    message: str
    model: str = "logreg"  # 'logreg' or 'nb'


app = FastAPI(title="SMS Spam Classifier")


def load_artifacts(model_dir: Path = None):
    if model_dir is None:
        model_dir = Path(__file__).resolve().parent.parent / "models"
    vec_path = model_dir / "vectorizer.joblib"
    nb_path = model_dir / "nb_model.joblib"
    logreg_path = model_dir / "logreg_model.joblib"
    if not vec_path.exists() or not nb_path.exists() or not logreg_path.exists():
        raise RuntimeError("Model artifacts not found. Run training first.")
    vect = joblib.load(vec_path)
    nb = joblib.load(nb_path)
    logreg = joblib.load(logreg_path)
    return vect, {"nb": nb, "logreg": logreg}


@app.on_event("startup")
def startup_load():
    try:
        app.state.vectorizer, app.state.models = load_artifacts()
    except Exception as e:
        # Keep app running but raise when predict called
        app.state.vectorizer = None
        app.state.models = {}
        app.state.load_error = str(e)


@app.post("/predict")
def predict(req: PredictRequest):
    if app.state.vectorizer is None:
        raise HTTPException(status_code=500, detail=app.state.load_error)
    if req.model not in app.state.models:
        raise HTTPException(status_code=400, detail="model must be 'logreg' or 'nb'")
    cleaned = clean_text(req.message)
    X = app.state.vectorizer.transform([cleaned])
    model = app.state.models[req.model]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        # classes_ ordering
        classes = model.classes_
        # find probability for predicted class
        pred_idx = probs.argmax()
        pred_label = classes[pred_idx]
        confidence = float(probs[pred_idx])
    else:
        pred_label = model.predict(X)[0]
        confidence = 1.0
    return {"label": pred_label, "confidence": round(confidence, 4)}

@app.get("/")
def home():
    return {"message": "Spam Detection API running"}



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
