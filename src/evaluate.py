from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from preprocessing import clean_text

print("Starting evaluation...")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "data.csv"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# Load dataset
df = pd.read_csv(DATA_PATH, encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['clean'] = df['message'].apply(clean_text)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# Load model
vectorizer = joblib.load(MODEL_DIR / "vectorizer.joblib")
model = joblib.load(MODEL_DIR / "logreg_model.joblib")

# Predict
X_test_vec = vectorizer.transform(X_test)
preds = model.predict(X_test_vec)

# Metrics
acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, pos_label="spam")
rec = recall_score(y_test, preds, pos_label="spam")
f1 = f1_score(y_test, preds, pos_label="spam")

print("\nModel Evaluation:")
print(f"Accuracy  : {acc:.2f}")
print(f"Precision : {prec:.2f}")
print(f"Recall    : {rec:.2f}")
print(f"F1-score  : {f1:.3f}")
