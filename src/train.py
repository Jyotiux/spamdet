from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from preprocessing import clean_text

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "data.csv"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("Loading data...")

df = pd.read_csv(DATA_PATH, encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['clean'] = df['message'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['clean'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print("Training TF-IDF...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),     # big improvement
    max_features=5000,
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)

print("Training Logistic Regression...")

model = LogisticRegression(
    class_weight="balanced",  # critical for spam
    max_iter=2000
)

model.fit(X_train_vec, y_train)

joblib.dump(vectorizer, MODEL_DIR / "vectorizer.joblib")
joblib.dump(model, MODEL_DIR / "logreg_model.joblib")

print("Training complete. Models saved.")
