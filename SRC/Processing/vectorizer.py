import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def create_tfidf_vectors():
    # đọc dataset sạch
    df = pd.read_csv(
        "Data/Processed/cleaned_resume.csv"
    )
    # xóa dòng rỗng
    df = df.dropna(subset=["cleaned_resume"])
    # lấy text
    resumes = df["cleaned_resume"]
    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=500
    )
    X = vectorizer.fit_transform(resumes)
    # dataframe vector
    vectors_df = pd.DataFrame(
        X.toarray()
    )
    # save vector
    vectors_df.to_csv(
        "Data/Processed/resume_vectors.csv",
        index=False
    )
    # save vectorizer
    with open(
        "Data/Processed/tfidf_vectorizer.pkl",
        "wb"
    ) as f:
        pickle.dump(vectorizer, f)
    print("TF-IDF completed!")
    print(vectors_df.head())
    return X