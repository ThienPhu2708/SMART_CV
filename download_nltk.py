import ssl
import nltk

ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
