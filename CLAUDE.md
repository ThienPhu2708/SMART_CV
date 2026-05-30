# SmartCV — Đồ án Deep Learning (HUIT)
**Đề tài:** Thiết kế hệ thống đánh giá ứng viên đa tầng: Kết hợp mô hình ngôn ngữ ngữ cảnh (BERT), mạng nơ-ron (MLP) và cổng Logic  
**GVHD:** Nguyễn Thanh Long  
**Nhóm 14:**
- Phùng Dương Thiên Phú – 2001230673
- Nguyễn Thị Minh Thư – 2001230959
- Nguyễn Thị Mỷ Duyên

---

## 1. Tổng quan hệ thống

Hệ thống sàng lọc CV tự động theo kiến trúc **4 tầng**:

```
Tầng 1: Trích xuất văn bản   PDF/JPG/PNG → pdfplumber → PyPDF2 → EasyOCR
Tầng 2: BERT Encoder (frozen) văn bản → paraphrase-multilingual-MiniLM-L12-v2 → 384-dim
Tầng 3: MLP Head              384 → 256 → 128 → 6 ngành (Softmax)
Tầng 4: Logic Gates           AND / OR / NOT / XOR → ĐẠT / LOẠI
```

**Chạy ứng dụng:**
```powershell
.\venv\Scripts\Activate.ps1
python run_app.py   # → http://localhost:8000
```

---

## 2. Dataset

| Thuộc tính | Giá trị |
|---|---|
| Nguồn gốc | Kaggle UpdatedResumeDataSet + Neuralframe + Synthetic |
| Tổng mẫu | ~1.197 CV / 6 ngành |
| Cân bằng | Tỉ lệ max/min = 1.03x |
| Ngôn ngữ | Tiếng Anh (có hỗ trợ dịch tiếng Việt) |

**6 ngành phân loại:**
```
INFORMATION-TECHNOLOGY | BUSINESS-DEVELOPMENT | SALES
DIGITAL-MEDIA | PUBLIC-RELATIONS | CONSULTANT
```

**Phân chia dữ liệu:**
```
Train : 70%  — huấn luyện trọng số
Val   : 15%  — Early Stopping, chọn model tốt nhất
Test  : 15%  — đánh giá cuối, chỉ dùng 1 lần (unbiased)
```

---

## 3. Kiến trúc Model

### BERT Encoder (Tầng 2 — frozen)
- **Model:** `paraphrase-multilingual-MiniLM-L12-v2`
- **Output:** 384 chiều (KHÔNG phải 768 — MiniLM nén từ BERT-large)
- **Hỗ trợ:** Tiếng Anh + tiếng Việt (multilingual)
- **Cache:** `Data/Processed/bert_embeddings.npy`

### MLP Head (Tầng 3 — trained)
```
Linear(384→256) → BatchNorm1d → ReLU → Dropout(0.45)
Linear(256→128) → BatchNorm1d → ReLU → Dropout(0.35)
Linear(128→6)   → Softmax
```

**Hyperparameters training:**
```
Epochs         : 80 (Early stopping patience=15)
Batch size     : 32
Learning rate  : 2e-4
Optimizer      : AdamW (weight_decay=8e-3)
Scheduler      : ReduceLROnPlateau (factor=0.5, patience=5)
Loss           : CrossEntropyLoss + LabelSmoothing(0.1) + Class Weights
```

> **Tại sao ReduceLROnPlateau thay CosineAnnealingLR:**
> CosineAnnealing gây dao động Val F1 ±3%. ReduceLROnPlateau chỉ giảm LR khi plateau → hội tụ mượt hơn.

### Kết quả

| Chỉ số | Giá trị |
|---|---|
| Best Val F1 | 91.54% (epoch ~42) |
| Best Val Accuracy | 91.62% |
| Test Accuracy | **88.89%** |
| Test Macro F1 | **88.84%** |
| Overfitting gap | ~2.1% (Train 91% vs Test 88.9%) |
| Out-of-sample (60 CV) | **76.7%** |

---

## 4. MLP Baseline (so sánh — không dùng trong production)

**File:** `SRC/model_mlp.py`, train bằng `train.py`

```
Input (600) → Linear(600→64) → BatchNorm1d → ReLU → Dropout(0.4)
            → Linear(64→32)  → BatchNorm1d → ReLU → Dropout(0.3)
            → Linear(32→6)   → Softmax
```

| Chỉ số | Giá trị |
|---|---|
| Best Val F1 | 87.73% (epoch 17) |
| Test Accuracy | 85.56% |
| Overfitting gap | ~13.4% (Train 99% vs Test 85.6%) |

> Giữ lại để minh chứng cải thiện khi thay TF-IDF bằng BERT encoding.

---

## 5. So sánh hai phương pháp (cùng Test Set 15%)

| Chỉ số | MLP Baseline (TF-IDF) | BERT + MLP (pipeline chính) |
|---|---|---|
| Test Accuracy | 85.56% | **88.89%** |
| Test Macro F1 | 85.67% | **88.84%** |
| Overfitting gap | ~13.4% | **~2.1%** |
| Embedding | Bag-of-words (600 dim) | Contextual (384 dim) |

---

## 6. Cấu trúc thư mục quan trọng

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
│   ├── model_bert.py             # MLP classification head (384→256→128→6)
│   ├── model_mlp.py              # MLP baseline (600→64→32→6)
│   ├── logic_gates.py            # AND/OR/NOT/XOR
│   └── Processing/
│       ├── cleaner.py            # clean_text(), translate_if_needed(), _has_vietnamese()
│       ├── vectorizer.py         # TF-IDF (baseline only)
│       ├── extractor.py          # PDF/OCR extraction
│       └── bert_encoder.py       # encode_texts(), encode_single()
│
├── Models/
│   ├── bert_classifier.pth       # BERT+MLP weights (model chính thức)
│   ├── bert_config.json          # Config + test metrics
│   ├── smartcv_model.pth         # MLP baseline weights
│   ├── classes.npy               # Label classes
│   ├── bert_training_curves.png
│   ├── training_curves.png       # MLP baseline curves
│   ├── confusion_matrix.png      # Confusion matrix (BERT)
│   ├── per_class_metrics.png
│   ├── roc_curves.png
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
│   │   ├── current_job.json
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
├── train_bert.py                 # Train BERT + MLP head (pipeline chính)
├── train.py                      # Train MLP baseline
├── main.py                       # Tiền xử lý + TF-IDF vectorize
├── predict.py                    # CLI inference (4-layer pipeline)
├── eval_test_cvs.py              # Đánh giá out-of-sample trên Test_CVs/
├── run_app.py                    # Khởi động web app
└── requirements.txt
```

---

## 7. Pipeline xử lý CV (khi upload)

```
Upload file (PDF / JPG / PNG)
    ↓
Tầng 1 — Trích xuất text:
  PDF text  → pdfplumber → PyPDF2 → EasyOCR (fallback scanned)
  JPG/PNG   → EasyOCR (ảnh gốc + ảnh preprocessed, ghép kết quả)
    ↓
Phát hiện ngôn ngữ:
  Tỉ lệ ký tự tiếng Việt > 8%? → dịch sang tiếng Anh
  Không? → langdetect
    ↓
Tiền xử lý (cleaner.py):
  - Boost Skills section (nhân đôi nội dung phần Skills)
  - Lowercase, xóa URL/email/phone
  - Chuẩn hóa viết tắt (ML→machine learning, AI→artificial intelligence...)
  - Lemmatization + remove stop words
    ↓
Tầng 2 — BERT Encoder:
  encode_single(text) → 384-dim embedding
    ↓
Tầng 3 — MLP Head:
  384 → 256 → 128 → 6 → Softmax → confidence%
    ↓
Tầng 4 — Logic Gates:
  AND / OR / NOT / XOR
    ↓
Kết quả: ĐẠT / LOẠI + confidence% + ngành + logic_score
```

---

## 8. Logic Gates — Cách hoạt động

| Gate | Ý nghĩa | Ví dụ |
|---|---|---|
| AND | Bắt buộc TẤT CẢ | Python AND SQL AND Docker |
| OR | Ít nhất 1 trong nhiều | ReactJS OR VueJS OR Angular |
| NOT | Loại nếu chứa | NOT "fresher" NOT "intern" |
| XOR | Đúng 1 trong 2 | Frontend XOR Backend |

Logic là **advisory** — BERT+MLP luôn chạy, logic chỉ quyết định ĐẠT/LOẠI.
Nếu LOẠI nhưng confidence ≥ 30% → hiển thị cảnh báo khuyên HR xem lại.

---

## 9. Tiền xử lý đặc biệt

**Phát hiện tiếng Việt — ngưỡng 8% (cleaner.py):**
```python
def _has_vietnamese(text) -> bool:
    vi_chars = len(_VI_CHARS.findall(text))
    alpha_chars = sum(1 for c in text if c.isalpha())
    return (vi_chars / alpha_chars) > 0.08
# Ngưỡng 8% tránh dịch CV tiếng Anh chỉ có tên/địa chỉ tiếng Việt
```

**Skills boosting (cleaner.py):**
```python
def _boost_skills_section(text, repeat=2)
# Nhân đôi nội dung phần Skills → từ kỹ năng có trọng số cao hơn
```

---

## 10. Đánh giá out-of-sample (Test_CVs)

**Script:** `eval_test_cvs.py`  
**Ground truth:** Tự động từ tên subfolder trong `6_nganh_chinh/`

| Ngành | Kết quả | Accuracy |
|---|---|---|
| BUSINESS-DEVELOPMENT | 10/10 | 100% |
| INFORMATION-TECHNOLOGY | 9/10 | 90% |
| DIGITAL-MEDIA | 9/10 | 90% |
| PUBLIC-RELATIONS | 7/10 | 70% |
| SALES | 6/10 | 60% |
| CONSULTANT | 5/10 | 50% |
| **Tổng** | **46/60** | **76.7%** |

**Thêm CV test:** Chỉ cần bỏ file vào đúng subfolder, script tự nhận ground truth.

---

## 11. Câu hỏi thường gặp khi bảo vệ đồ án

**"Tại sao dùng BERT thay TF-IDF?"**
> TF-IDF là bag-of-words — không phân biệt ngữ cảnh. CV Frontend Developer có từ "SEO" bị nhầm là Digital-Media. BERT hiểu ngữ cảnh: "SEO" trong Skills của người dùng TypeScript/Next.js → IT.

**"BERT có vi phạm yêu cầu đề tài chỉ dùng MLP không?"**
> Không. Classification head là MLP thuần túy (384→256→128→6). Encoder chỉ thay thế TF-IDF ở bước trích xuất đặc trưng.

**"Embedding dim của BERT là bao nhiêu?"**
> 384 chiều. paraphrase-multilingual-MiniLM-L12-v2 là phiên bản nén, xuất 384-dim thay vì 768-dim của BERT-base gốc.

**"Tại sao CONSULTANT accuracy thấp nhất (50%) trong test thực tế?"**
> CV tư vấn đề cập nhiều lĩnh vực (IT, kinh doanh, quản lý) → ranh giới với BD và IT mờ. Hạn chế do dữ liệu ~200 mẫu/ngành còn nhỏ.

**"Tại sao cả hai model dùng cùng tỉ lệ chia?"**
> Để so sánh công bằng (apples-to-apples). Nếu MLP test trên 15% còn BERT test trên 20%, kết quả không có giá trị thuyết phục.

---

## 12. Lệnh chạy các bước chính

```powershell
# Kích hoạt môi trường
.\venv\Scripts\Activate.ps1

# Train BERT + MLP (pipeline chính thức)
python train_bert.py

# Train MLP Baseline (so sánh — tùy chọn)
python main.py
python train.py

# Đánh giá out-of-sample
python eval_test_cvs.py

# Khởi động web app
python run_app.py   # → http://localhost:8000
```

---

## 13. Requirements chính

```
torch>=2.0.0
scikit-learn>=1.2.0
sentence-transformers>=2.2.0   # BERT encoder
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
pdfplumber>=0.10.0
easyocr>=1.7.0
deep-translator>=1.11.0
langdetect>=1.0.9
```

---

## 14. Lưu ý quan trọng

> ⚠️ **Nếu chạy lại `main.py`, phải chạy lại `train.py` ngay sau.**
> Thứ tự đúng: `main.py` → `train.py`

> ⚠️ **Thêm CV vào Test_CVs/6_nganh_chinh/:** Bỏ file vào đúng subfolder.
> `eval_test_cvs.py` tự nhận ground truth từ tên thư mục, không cần cập nhật code.
