import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

def create_tfidf_vectors():
    print("Dang bat dau Vector hoa nang cao (2000 dac trung)...")
    
    # 1. Đọc dữ liệu đã làm sạch từ Thư
    path_cleaned = "Data/Processed/cleaned_resume.csv"
    if not os.path.exists(path_cleaned):
        print("Loi: Khong tim thay file cleaned_resume.csv. Hay chay buoc lam sach truoc!")
        return
        
    df = pd.read_csv(path_cleaned)
    df = df.dropna(subset=["cleaned_resume"])
    resumes = df["cleaned_resume"]

    # 2. TỐI ƯU: Tăng lên 2000 đặc trưng và lọc nhiễu chuyên sâu
    # min_df=2: Loại bỏ từ chỉ xuất hiện 1 lần (thường là tên riêng hoặc lỗi đánh máy)
    # max_df=0.8: Loại bỏ các từ xuất hiện ở hơn 80% số lượng hồ sơ (từ quá phổ thông)
    vectorizer = TfidfVectorizer(
        max_features=800,    # giảm từ 1500 → 800: tỷ lệ features/samples tốt hơn
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.75,
        stop_words='english'
    )
    
    X = vectorizer.fit_transform(resumes)
    
    # 3. Lưu ma trận vector dưới dạng CSV cho Phú luyện AI
    vectors_df = pd.DataFrame(X.toarray())
    vectors_df.to_csv("Data/Processed/resume_vectors.csv", index=False)
    
    # 4. Lưu vectorizer mẫu để dùng cho file predict.py sau này
    with open("Data/Processed/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
        
    print(f"TF-IDF hoan tat! Kich thuoc ma tran: {X.shape}")
    return X

if __name__ == "__main__":
    # Đảm bảo thư mục tồn tại
    os.makedirs("Data/Processed", exist_ok=True)
    create_tfidf_vectors()