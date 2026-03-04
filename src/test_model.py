import joblib

# Load model
model = joblib.load("models/svm_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_email(text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return prediction[0]

if __name__ == "__main__":
    while True:
        email = input("Enter email text: ")
        result = predict_email(email)

        if result == 1:
            print("Prediction: SPAM 🚨")
        else:
            print("Prediction: NOT SPAM ✅")