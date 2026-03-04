from flask import Flask, render_template, request
import joblib
import json
import sys
import os

# Allow access to src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import clean_text

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("models/svm_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Load metrics
with open("models/metrics.json") as f:
    metrics = json.load(f)

accuracy = metrics["accuracy"]


# Function to extract important keywords
def get_top_keywords(text, vectorizer, model, top_n=5):

    words = text.split()
    word_scores = []

    for word in words:
        if word in vectorizer.vocabulary_:

            index = vectorizer.vocabulary_[word]
            weight = model.coef_[0][index]

            word_scores.append((word, weight))

    word_scores = sorted(word_scores, key=lambda x: abs(x[1]), reverse=True)

    return [w[0] for w in word_scores[:top_n]]


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    email_text = ""
    prediction_class = ""

    if request.method == "POST":

        email_text = request.form["email"]

        cleaned = clean_text(email_text)
        vectorized = vectorizer.transform([cleaned])

        result = model.predict(vectorized)[0]

        score = model.decision_function(vectorized)[0]
        confidence = abs(score)

        if result == 1:
            prediction = f"SPAM 🚨 (confidence: {confidence:.2f})"
            prediction_class = "spam"
        else:
            prediction = f"NOT SPAM ✅ (confidence: {confidence:.2f})"
            prediction_class = "ham"

    return render_template(
        "index.html",
        prediction=prediction,
        prediction_class=prediction_class,
        accuracy=accuracy,
        email_text=email_text
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)