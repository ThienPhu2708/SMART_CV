import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def clean_text(text):
    # lowercase
    text = text.lower()
    # remove email
    text = re.sub(r'\S+@\S+', '', text)
    # remove url
    text = re.sub(r'http\S+|www\S+', '', text)
    # remove phone numbers
    text = re.sub(r'\d+', '', text)
    # remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # remove stopwords
    words = text.split()
    filtered_words = [
        word for word in words
        if word not in stop_words
    ]
    return " ".join(filtered_words)