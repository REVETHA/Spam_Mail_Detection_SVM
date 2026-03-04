import joblib
import re
import numpy as np
from scipy.sparse import hstack
import os

from data_preprocessing import clean_text

# Get absolute path to project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "hybrid_svm_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

# Load model safely
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def extract_metadata(text):
    num_urls = len(re.findall(r'http\S+', text))
    num_phone_numbers = len(re.findall(r'\b\d{10}\b', text))
    has_attachments = 0
    contains_tracking_token = 0

    return np.array([[num_urls,
                      num_phone_numbers,
                      has_attachments,
                      contains_tracking_token]], dtype=float)


def predict_email(text):
    cleaned = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned])

    metadata = extract_metadata(text)

    combined = hstack([text_tfidf, metadata])

    prediction = model.predict(combined)[0]

    return prediction