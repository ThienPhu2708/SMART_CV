# SmartCV — Hệ thống Đánh giá Ứng viên Đa tầng

Đồ án môn Deep Learning — Trường Đại học Công nghiệp TP.HCM (HUIT)

**Nhóm 14:**
| Họ tên | MSSV |
|---|---|
| Phùng Dương Thiên Phú | 2001230673 |
| Nguyễn Thị Minh Thư | 2001230959 |
| Nguyễn Thị Mỷ Duyên | 2001230130 |

**GVHD:** TS. Nguyễn Thanh Long

---

## Giới thiệu

SmartCV tự động hóa giai đoạn sơ loại hồ sơ tuyển dụng bằng kiến trúc **4 tầng** kết hợp BERT, MLP và Logic Gates:

```
Tầng 1 — Trích xuất     PDF / JPG / PNG  →  pdfplumber → PyPDF2 → EasyOCR (fallback scan)
Tầng 2 — BERT Encoder   văn bản          →  paraphrase-multilingual-MiniLM-L12-v2  →  384-dim
Tầng 3 — MLP Head       384-dim          →  256 → 128 → 6 ngành  →  Softmax confidence%
Tầng 4 — Logic Gates    AND / OR / NOT / XOR  →  ĐẠT / LOẠI
```

Hệ thống đọc được PDF có text, PDF scan, ảnh JPG/PNG, và CV tiếng Việt (tự động dịch sang tiếng Anh trước khi xử lý).

---

## Kết quả mô hình

### So sánh hai phương pháp (cùng tập Test 15%)

| Chỉ số | MLP Baseline (TF-IDF) | BERT + MLP (chính thức) | Cải thiện |
|---|---|---|---|
| Val Accuracy | 87.71% | **91.62%** | +3.91% |
| Val Macro F1 | 87.73% | **91.54%** | +3.81% |
| Test Accuracy | 85.56% | **88.89%** | +3.33% |
| Test Macro F1 | 85.67% | **88.44%** | +2.77% |
| Overfit gap | ~13.4% ⚠ | **~2.1% ✓** | −11.3% |
| Epoch thực tế | 17 | ~42 | — |

### Đánh giá out-of-sample (60 CV thực tế, ngoài tập huấn luyện)

| Ngành | Đúng / Tổng | Accuracy |
|---|---|---|
| BUSINESS-DEVELOPMENT | 10 / 10 | 100% |
| INFORMATION-TECHNOLOGY | 9 / 10 | 90% |
| DIGITAL-MEDIA | 9 / 10 | 90% |
| PUBLIC-RELATIONS | 7 / 10 | 70% |
| SALES | 6 / 10 | 60% |
| CONSULTANT | 5 / 10 | 50% |
| **Tổng** | **46 / 60** | **76.7%** |

---

## Cài đặt

**Yêu cầu:** Python 3.10+, RAM ≥ 4GB (không cần GPU)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Lần đầu chạy, BERT encoder (~120MB) được tải tự động từ Hugging Face.

---

## Chạy ứng dụng

```powershell
.\venv\Scripts\Activate.ps1
python run_app.py
# → http://localhost:8000
```

---

## Huấn luyện lại model

```powershell
# Train BERT + MLP (pipeline chính thức) — ~15–30 phút CPU
python train_bert.py

# Train MLP Baseline TF-IDF (để so sánh — tùy chọn)
python main.py
python train.py

# Đánh giá out-of-sample trên 60 CV thực tế
python eval_test_cvs.py
```

> **Lưu ý:** Nếu chạy lại `main.py`, phải chạy lại `train.py` ngay sau.

---

## Dataset

| Nguồn | Số mẫu | Ghi chú |
|---|---|---|
| Kaggle UpdatedResumeDataSet | 678 | CV tiếng Anh thực tế, lọc 6 ngành mục tiêu |
| Neuralframe Dataset | 284 | Bổ sung tối đa 80 mẫu/ngành |
| Synthetic (tự sinh) | 235 | Cân bằng ngành thiếu (DIGITAL-MEDIA, CONSULTANT) |
| **Tổng** | **~1.197** | Tỉ lệ max/min = 1.03x — gần cân bằng tuyệt đối |

**Phân chia:** 70% Train / 15% Val / 15% Test — Stratified Split, random seed 42

**6 ngành:** `INFORMATION-TECHNOLOGY` · `BUSINESS-DEVELOPMENT` · `SALES` · `DIGITAL-MEDIA` · `PUBLIC-RELATIONS` · `CONSULTANT`

---

## Kiến trúc MLP Head (Tầng 3)

```
Input:   384-dim  ← vector từ BERT Encoder
         ↓
Layer 1: Linear(384 → 256) → BatchNorm1d → ReLU → Dropout(0.45)
         ↓
Layer 2: Linear(256 → 128) → BatchNorm1d → ReLU → Dropout(0.35)
         ↓
Output:  Linear(128 → 6)  → Softmax → confidence%
```

**Hyperparameters:** AdamW (lr=2e-4, weight_decay=8e-3) · ReduceLROnPlateau · Early stopping patience=15 · LabelSmoothing=0.1

---

## Logic Gates (Tầng 4)

| Gate | Điều kiện | Ví dụ |
|---|---|---|
| AND | Bắt buộc TẤT CẢ kỹ năng | Python AND SQL AND Docker |
| OR | Ít nhất 1 kỹ năng trong danh sách | ReactJS OR VueJS OR Angular |
| NOT | Loại nếu chứa từ khóa cấm | NOT "fresher" NOT "intern" |
| XOR | Đúng 1 trong 2 điều kiện | Frontend XOR Backend |

Logic là **advisory**: BERT+MLP luôn chạy, logic chỉ ra kết quả ĐẠT/LOẠI. Nếu LOẠI nhưng confidence ≥ 30% → cảnh báo HR xem xét thủ công.

---

## Cấu trúc thư mục

```
SMART_CV/
├── App/
│   ├── api.py                    # FastAPI backend — tất cả routes
│   ├── static/
│   │   ├── css/app.css
│   │   └── js/app.js
│   └── templates/
│       ├── base.html
│       ├── dashboard.html
│       ├── upload.html
│       ├── results.html
│       ├── candidate_detail.html
│       ├── history.html
│       ├── settings.html
│       └── jobs/new.html
│
├── SRC/
│   ├── model_bert.py             # MLP head (384 → 256 → 128 → 6)
│   ├── model_mlp.py              # MLP baseline (600 → 64 → 32 → 6)
│   ├── logic_gates.py            # AND / OR / NOT / XOR evaluation
│   └── Processing/
│       ├── cleaner.py            # clean_text(), translate_if_needed(), _has_vietnamese()
│       ├── vectorizer.py         # TF-IDF (baseline only)
│       ├── extractor.py          # PDF / OCR extraction
│       └── bert_encoder.py       # encode_texts(), encode_single()
│
├── Models/
│   ├── bert_classifier.pth       # Weights BERT + MLP (model chính thức)
│   ├── bert_config.json          # Config + test metrics
│   ├── smartcv_model.pth         # Weights MLP baseline
│   ├── classes.npy               # Label encoder classes
│   ├── bert_training_curves.png  # Training curves BERT + MLP
│   ├── training_curves.png       # Training curves MLP baseline
│   ├── confusion_matrix.png      # Confusion matrix (BERT + MLP)
│   ├── per_class_metrics.png     # F1 / Precision / Recall từng ngành
│   ├── roc_curves.png            # ROC curves 6 ngành
│   └── test_evaluation/
│       ├── test_results.csv
│       ├── test_results.json
│       └── test_evaluation_report.png
│
├── Data/
│   ├── Processed/
│   │   ├── cleaned_resume.csv
│   │   ├── bert_embeddings.npy   # BERT embeddings cache (384-dim)
│   │   ├── bert_labels.npy
│   │   ├── tfidf_vectorizer.pkl
│   │   ├── current_job.json      # Job đang tuyển dụng
│   │   └── screening_history.json
│   └── Test_CVs/
│       ├── 6_nganh_chinh/        # 60 CV thực tế (10 CV × 6 ngành)
│       │   ├── INFORMATION-TECHNOLOGY/
│       │   ├── BUSINESS-DEVELOPMENT/
│       │   ├── SALES/
│       │   ├── DIGITAL-MEDIA/
│       │   ├── PUBLIC-RELATIONS/
│       │   └── CONSULTANT/
│       ├── ocr_test/             # CV dạng ảnh scan
│       ├── tieng_viet/           # CV tiếng Việt
│       └── ngoai_he_thong/       # CV ngoài 6 ngành
│
├── train_bert.py                 # Train BERT + MLP (pipeline chính thức)
├── train.py                      # Train MLP baseline (so sánh)
├── main.py                       # Tiền xử lý + TF-IDF vectorize
├── predict.py                    # CLI inference (4-layer pipeline)
├── eval_test_cvs.py              # Đánh giá out-of-sample trên Test_CVs/
├── run_app.py                    # Khởi động web app
└── requirements.txt
```

> **Thêm CV test:** Bỏ file vào đúng subfolder trong `Test_CVs/6_nganh_chinh/`. Script `eval_test_cvs.py` tự nhận ground truth từ tên thư mục.

---

## Công nghệ

| Nhóm | Thư viện |
|---|---|
| Deep Learning | PyTorch ≥ 2.0 |
| BERT Encoder | sentence-transformers ≥ 2.2 |
| ML / Đánh giá | scikit-learn ≥ 1.2, imbalanced-learn |
| NLP | NLTK |
| PDF | pdfplumber ≥ 0.10, PyPDF2, PyMuPDF |
| OCR | EasyOCR ≥ 1.7, OpenCV, Pillow |
| Dịch thuật | deep-translator (GoogleTranslator) |
| Phát hiện ngôn ngữ | langdetect |
| Web Backend | FastAPI ≥ 0.110, Uvicorn, Jinja2 |
| Visualization | matplotlib, seaborn |
