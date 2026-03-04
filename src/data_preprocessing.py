import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def clean_text(text):
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra spaces
    text = text.strip()

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Combine subject + body
    df['text'] = df['subject'] + " " + df['body_plain']

    # Clean text
    df['cleaned_text'] = df['text'].apply(clean_text)

    return df


if __name__ == "__main__":
    data_path = "data/spam_mail_data.csv"
    df = load_and_preprocess(data_path)

    print(df[['cleaned_text', 'label']].head())