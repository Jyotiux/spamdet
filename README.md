# Message Spam Classifier

This project trains models to classify SMS messages as spam or ham, and exposes a FastAPI backend for predictions.

Structure:
- `src/preprocessing.py` - text cleaning utilities
- `src/train.py` - download dataset, train TF-IDF + Naive Bayes and Logistic Regression, save models
- `src/evaluate.py` - evaluation metrics (accuracy, precision, recall, confusion matrix)
- `src/api.py` - FastAPI app with `/predict` endpoint

Quick start:

1. Create a virtualenv and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Train models (this downloads the SMSSpamCollection dataset):

```bash
python -m src.train
```

3. Run FastAPI server:

```bash
uvicorn src.api:app --reload
```

4. Predict (example):

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"message\": \"Free entry in 2 a wkly comp to win FA Cup final tkts\"}"
```

Notes:
- Preprocessing: lowercasing, punctuation removal, stopword removal using `sklearn` stopwords.
- Vectorizer and models are saved to `models/` after training.
