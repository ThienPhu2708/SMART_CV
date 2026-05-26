# SmartCV — Hệ thống Sàng lọc Hồ sơ Tự động

Dự án cuối kỳ môn Deep Learning — HUIT  
Kết hợp **MLP Neural Network** + **Logic Gates** + **OCR** để phân loại và lọc CV ứng viên tự động.

---

## Giới thiệu

SmartCV là hệ thống sàng lọc hồ sơ (CV) thông minh dành cho bộ phận HR. Thay vì đọc từng CV thủ công, hệ thống tự động:

1. Trích xuất văn bản từ file PDF (kể cả CV scan/ảnh qua OCR)
2. Dịch tự động nếu CV không phải tiếng Anh (hỗ trợ tiếng Việt và nhiều ngôn ngữ khác)
3. Phân loại ngành nghề bằng mạng nơ-ron MLP
4. Lọc ứng viên qua bộ Logic Gates tùy chỉnh (AND / OR / NOT / XOR)
5. Hiển thị kết quả và biểu đồ trực quan qua giao diện Streamlit

---

## Kiến trúc hệ thống

```
PDF Upload
    │
    ▼
[Extractor] ── pdfplumber / PyPDF2 / EasyOCR (fallback)
    │
    ▼
[Translator] ── langdetect + Google Translate (nếu không phải tiếng Anh)
    │
    ▼
[Cleaner / NLP] ── lowercase, remove noise, lemmatization (NLTK WordNet)
    │
    ├──► [Logic Gates] ── AND / OR / NOT / XOR  →  advisory filter
    │
    └──► [MLP Classifier] ── TF-IDF (800 features) → 256 → 128 → 64 → 24 classes
              │
              ▼
         [Kết quả] ── Ngành nghề + Độ tin cậy + Trạng thái ĐẠT/LOẠI
```

> **Nguyên tắc thiết kế:** Logic Gates đóng vai trò cố vấn (advisory), MLP **luôn chạy** bất kể kết quả lọc. Nếu Logic Gate loại nhưng MLP nhận diện đúng ngành với độ tin cậy cao, hệ thống sẽ cảnh báo HR xem xét lại tiêu chí.

---

## Cấu trúc thư mục

```
SMART_CV/
│
├── App/
│   └── main_ui.py              # Giao diện Streamlit (6 trang)
│
├── Data/
│   ├── Raw/
│   │   ├── Resume.csv              # Dataset gốc (~2484 mẫu, 24 ngành)
│   │   ├── neuralframe_resumes.csv # Dataset bổ sung (NF_ prefix)
│   │   ├── synthetic_resumes.csv   # Dữ liệu tổng hợp (SYN_ prefix)
│   │   └── Resume_merged.csv       # Dataset gộp (gốc + NF)
│   ├── Processed/
│   │   ├── cleaned_resume.csv      # Văn bản sau khi làm sạch NLP
│   │   ├── resume_vectors.csv      # Vector TF-IDF
│   │   ├── tfidf_vectorizer.pkl    # Vectorizer đã fit
│   │   └── screening_history.json  # Lịch sử sàng lọc CV
│   └── Test_CVs/                   # 24 CV mẫu PDF (1 CV/ngành)
│
├── Models/
│   ├── smartcv_model.pth       # Trọng số MLP đã huấn luyện
│   ├── classes.npy             # Danh sách 24 nhãn ngành
│   ├── training_curves.png     # Biểu đồ Loss/Accuracy theo epoch
│   └── confusion_matrix.png    # Ma trận nhầm lẫn trên tập test
│
├── SRC/
│   ├── Processing/
│   │   ├── extractor.py            # Trích xuất PDF + OCR (EasyOCR)
│   │   ├── cleaner.py              # Làm sạch NLP + dịch ngôn ngữ
│   │   ├── vectorizer.py           # TF-IDF vectorization
│   │   ├── merge_datasets.py       # Gộp các dataset nguồn
│   │   └── synthetic_generator.py  # Tạo dữ liệu tổng hợp
│   ├── model_mlp.py            # Định nghĩa kiến trúc MLP
│   └── logic_gates.py          # AND / OR / NOT / XOR gates
│
├── main.py         # Tiền xử lý dữ liệu + TF-IDF (chạy trước train)
├── train.py        # Huấn luyện MLP (SMOTE + Early Stopping)
├── predict.py      # Pipeline dự đoán từ PDF (CLI)
├── create_test_cvs.py  # Tạo 24 CV PDF mẫu để kiểm thử
├── download_nltk.py    # Tải NLTK data (stopwords, wordnet)
└── requirements.txt
```

---

## Dataset

| Nguồn | Ký hiệu | Số lượng | Ghi chú |
|---|---|---|---|
| Kaggle Resume Dataset | _(gốc)_ | ~2484 mẫu | 24 ngành nghề |
| Neuralframe Resumes | `NF_` prefix | tối đa 200/ngành | Bổ sung ngành thiếu |
| Synthetic Resumes | `SYN_` prefix | tự sinh | Cân bằng lớp thiểu số |

**24 ngành nghề được hỗ trợ:**  
Accountant, Advocate, Agriculture, Apparel, Arts, Automobile, Aviation, Banking, BPO, Business Development, Chef, Construction, Consultant, Designer, Digital Media, Engineering, Finance, Fitness, Healthcare, HR, Information Technology, Public Relations, Sales, Teacher

---

## Kiến trúc MLP

```
Input (TF-IDF, 800 features)
    │
    ├── Linear(800→256) → BatchNorm → ReLU → Dropout(0.6)
    ├── Linear(256→128) → BatchNorm → ReLU → Dropout(0.5)
    ├── Linear(128→64)  → BatchNorm → ReLU → Dropout(0.4)
    └── Linear(64→24)   → CrossEntropyLoss
```

- **Optimizer:** Adam, lr=1e-3, weight_decay=1e-3
- **Oversampling:** SMOTE (imblearn) trên tập train để cân bằng lớp
- **Early Stopping:** patience=12 epoch, monitor val_loss
- **Weight init:** He (Kaiming) initialization

---

## Logic Gates

| Gate | Điều kiện | Hành vi khi thất bại |
|---|---|---|
| **AND** | Tất cả kỹ năng bắt buộc phải có | Cảnh báo, ghi lý do |
| **OR** | Ít nhất 1 kỹ năng ưu tiên phải có | Ghi nhận thiếu ưu tiên |
| **NOT** | Không chứa từ khóa cấm | Cảnh báo, ghi lý do |
| **XOR** | Đúng 1 trong 2 kỹ năng độc quyền | Cảnh báo |

> Keyword matching hỗ trợ: exact match → lemmatized match → fuzzy match (Jaro-Winkler).

---

## Cài đặt

```bash
# Tạo môi trường ảo
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Cài thư viện
pip install -r requirements.txt

# Tải NLTK data
python download_nltk.py
```

---

## Cách chạy

### 1. Tiền xử lý dữ liệu
```bash
python main.py
```
Làm sạch văn bản, tạo TF-IDF vectors, lưu `tfidf_vectorizer.pkl`.

### 2. Huấn luyện mô hình
```bash
python train.py
```
Huấn luyện MLP với SMOTE + Early Stopping, lưu `smartcv_model.pth`, vẽ `training_curves.png` và `confusion_matrix.png`.

### 3. Chạy giao diện Streamlit
```bash
streamlit run App/main_ui.py
```

### 4. Dự đoán từ CLI (tùy chọn)
```bash
python predict.py
```
Chỉnh sửa `CV_PATH` và `REQUIRED_SKILLS` trong file trước khi chạy.

---

## Giao diện Streamlit (6 trang)

| Trang | Chức năng |
|---|---|
| 🏠 Trang chủ | Dashboard tổng quan, tỷ lệ ĐẠT/LOẠI, quy trình hoạt động |
| 💼 Chọn vị trí | Dropdown 24 vị trí, tự động điền kỹ năng mẫu |
| ⚙️ Cấu hình lọc | Chỉnh sửa AND/OR/NOT/XOR gates với tag màu + Add/Remove |
| 📤 Upload CV | Upload PDF, chạy toàn bộ pipeline, hiển thị kết quả nhanh |
| 📊 Kết quả & Biểu đồ | Gauge chart, Top-8 ngành, Logic Gate visual, Skill match chart |
| 📁 Lịch sử | Bảng lịch sử toàn bộ CV, biểu đồ phân bố ngành, xuất CSV |

---

## Công nghệ sử dụng

| Thành phần | Thư viện |
|---|---|
| Deep Learning | PyTorch |
| Xử lý dữ liệu | Pandas, NumPy, Scikit-learn |
| Oversampling | imbalanced-learn (SMOTE) |
| NLP | NLTK (stopwords, WordNetLemmatizer) |
| TF-IDF | Scikit-learn TfidfVectorizer |
| PDF extraction | pdfplumber, PyPDF2, PyMuPDF |
| OCR | EasyOCR, OpenCV, Pillow |
| Phát hiện ngôn ngữ | langdetect |
| Dịch tự động | deep-translator (Google Translate) |
| Giao diện | Streamlit |
| Biểu đồ | Matplotlib |

---

## Yêu cầu hệ thống

- Python 3.10+
- RAM: tối thiểu 4GB (EasyOCR cần ~1.5GB khi load)
- GPU: không bắt buộc (CPU inference đủ nhanh với dataset này)
