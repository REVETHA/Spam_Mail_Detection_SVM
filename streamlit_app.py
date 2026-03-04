import streamlit as st
import joblib
import json
import sys
import os
import matplotlib.image as mpimg

# Access src folder
sys.path.append(os.path.abspath("src"))

from data_preprocessing import clean_text

# Load model
model = joblib.load("models/svm_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Load metrics
with open("models/metrics.json") as f:
    metrics = json.load(f)

accuracy = metrics["accuracy"]

st.set_page_config(page_title="Spam Mail Detection", page_icon="📧")

st.title("📧 Spam Mail Detection (SVM)")
st.write(f"Model Accuracy: **{accuracy}**")

email_text = st.text_area("Paste email content here")

if st.button("Check Email"):

    cleaned = clean_text(email_text)
    vectorized = vectorizer.transform([cleaned])

    result = model.predict(vectorized)[0]

    score = model.decision_function(vectorized)[0]
    confidence = abs(score)

    if result == 1:
        st.error(f"SPAM 🚨 (confidence: {confidence:.2f})")
    else:
        st.success(f"NOT SPAM ✅ (confidence: {confidence:.2f})")

st.divider()

st.subheader("SVM Spam vs Ham Decision Boundary")

img = mpimg.imread("app/static/svm_plot.png")
st.image(img)

st.divider()

st.caption("Built by Revetha K | Spam Detection using Support Vector Machine")