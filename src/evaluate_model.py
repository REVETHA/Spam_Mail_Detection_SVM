import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from data_preprocessing import load_and_preprocess
from sklearn.model_selection import train_test_split


def evaluate(data_path):
    df = load_and_preprocess(data_path)

    X = df['cleaned_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = joblib.load("models/svm_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")

    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    evaluate("data/spam_mail_data.csv")