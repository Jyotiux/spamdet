# Message Spam Classifier 

A small project for detecting spam messages. It trains both **Logistic Regression** and **Multinomial Naive Bayes** using TF‑IDF features, provides evaluation utilities, a Streamlit demo UI, and a FastAPI prediction endpoint.

---

## Repository structure 
- `src/preprocessing.py` — text cleaning utilities used by training, evaluation, and inference.
- `src/train.py` — trains TF‑IDF, Logistic Regression and Naive Bayes on the project dataset (80/20 split) and saves artifacts to `models/`.
- `src/evaluate.py` — evaluates saved models. By default (no args) it evaluates the project dataset with an 80/20 split; use `--data <file>` to evaluate a full external CSV file (no split).
- `src/app.py` — Streamlit UI for interactive predictions.
- `src/api.py` — FastAPI app exposing `/predict` for programmatic use.
- `data/` — example datasets (project SMS dataset: `data.csv`, and extra test files).
- `models/` — output directory where `vectorizer.joblib`, `logreg_model.joblib`, and `nb_model.joblib` are stored.

---

## Data format & robustness 
- The code accepts datasets with columns named either `v1`/`v2` (original SMS format) or `label`/`message`.
- If those headers are missing, the script will assume the **first two columns** are `label` and `message` respectively.
- Labels are normalized (trim + lowercased), so `Spam`, `SPAM`, or ` spam ` are treated as `spam`.

---

## Quick setup & commands 
1. Create env & install:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Train (80/20 split) — trains LR + NB and prints test metrics:

```powershell
python src/train.py
```

3. Evaluate default project dataset (80/20):

```powershell
python src/evaluate.py
```

4. Evaluate an external dataset (evaluates the entire file):

```powershell
python src/evaluate.py --data path\to\your_dataset.csv
```

5. Run the Streamlit demo:

```powershell
streamlit run src/app.py
```

6. Run the API server:

```powershell
uvicorn src.api:app --reload
```

---

## Datasets used-
https://www.kaggle.com/datasets/meruvulikith/190k-spam-ham-email-dataset-for-classification

https://github.com/mohitgupta-1O1/Kaggle-SMS-Spam-Collection-Dataset-/blob/master/spam.csv


---

## Reproducible results — current evaluations 
These were produced with the saved models in `models/`.

**Project SMS dataset (`data/data.csv`) — 80/20 split**
- Logistic Regression: **Accuracy 0.982**, Precision 0.945, Recall 0.919, F1 0.932
- Naive Bayes: **Accuracy 0.967**, Precision 0.991, Recall 0.758, F1 0.859

**External email dataset (`data/spam_Emails_data.csv`) — full-file evaluation**
- Logistic Regression: **Accuracy 0.544**, Precision 0.547, Recall 0.203, F1 0.296
- Naive Bayes: **Accuracy 0.530**, Precision 0.629, Recall 0.016, F1 0.031

> **Interpretation:** The models perform very well on the SMS dataset used for training but **do not generalize** to the email dataset (domain shift). Reported metrics are reproducible using the commands above.

---

## Why these metrics? 
We report **Accuracy**, **Precision**, **Recall**, and **F1-score** because each metric highlights a different aspect of classifier performance:

- **Accuracy** — overall proportion of correct predictions. It is a good high-level indicator but can be misleading when classes are imbalanced.
- **Precision** (for the *spam* class) — the proportion of messages predicted as spam that are actually spam. High precision means fewer false positives (legitimate messages mislabeled as spam).
- **Recall** (for the *spam* class) — the proportion of actual spam messages that were detected. High recall means the model misses fewer spam messages (fewer false negatives).
- **F1-score** — the harmonic mean of precision and recall; useful when you want a single number that balances both.


---


## Limitations & suggested next steps 
- Expect **domain shift**: SMS-trained models may fail on emails or other message types. To fix, **retrain** or fine‑tune with representative email examples, or combine datasets.
- Improve feature engineering: URL and email detection, character n‑grams, or larger `max_features` can help.

---
