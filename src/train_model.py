import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import load_and_preprocess


def train_model(data_path):
    # Load and preprocess
    df = load_and_preprocess(data_path)

    X = df['cleaned_text']
    y = df['label']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train SVM
    model = LinearSVC()
    model.fit(X_train_tfidf, y_train)

    # Predictions
    y_pred = model.predict(X_test_tfidf)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, "models/svm_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    # Save metrics
    import json
    metrics = {"accuracy": accuracy}

    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f)

    print("Model, vectorizer, and metrics saved!")


if __name__ == "__main__":
    train_model("data/spam_mail_data.csv")