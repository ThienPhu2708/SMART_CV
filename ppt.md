# BỐ CỤC PPT — HỆ THỐNG ĐÁNH GIÁ ỨNG VIÊN ĐA TẦNG (41 slides)

---

## SLIDE 1: TRANG BÌA
- Tiêu đề lớn: **HỆ THỐNG ĐÁNH GIÁ ỨNG VIÊN ĐA TẦNG**
- Dòng phụ: Kết hợp BERT, Mạng MLP và Bộ lọc Logic Gates
- Nhóm 14 — Đồ án Deep Learning — HUIT
- GVHD: TS. Nguyễn Thanh Long

---

## SLIDE 2: THÀNH VIÊN NHÓM 14
*(3 card dọc, mỗi card có tên + MSSV)*
- Phùng Dương Thiên Phú — 2001230673
- Nguyễn Thị Minh Thư — 2001230959
- Lê Thị Mỷ Duyên — 2001230130

---

## SLIDE 3: CHAPTER HEADER — CHƯƠNG 1: GIỚI THIỆU
*(Slide nền đậm, chỉ có tiêu đề chương ở giữa)*

---

## SLIDE 4: BỐI CẢNH VÀ VẤN ĐỀ
**Tiêu đề:** BỐI CẢNH VÀ VẤN ĐỀ

- Thị trường lao động 4.0 bùng nổ → mỗi đợt tuyển dụng nhận hàng trăm hồ sơ
- HR phải đọc và đánh giá thủ công từng CV — tốn thời gian, dễ bỏ sót
- Đánh giá cảm tính, thiếu nhất quán → ứng viên tốt bị loại oan
- Keyword matching đơn giản: chỉ tìm từ khóa, không hiểu ngữ cảnh
  - Ví dụ: CV có từ "Python" nhưng không biết context là học hay làm thật
- **→ Làm thế nào tự động hóa giai đoạn sơ loại hồ sơ một cách thông minh?**

---

## SLIDE 5: GIẢI PHÁP ĐỀ XUẤT — PIPELINE 4 TẦNG
**Tiêu đề:** GIẢI PHÁP ĐỀ XUẤT

*(4 cột dọc song song, mỗi cột = 1 tầng)*

| TẦNG 1 | TẦNG 2 | TẦNG 3 | TẦNG 4 |
|---|---|---|---|
| **TRÍCH XUẤT VĂN BẢN** | **MÃ HÓA NGỮ NGHĨA** | **MLP PHÂN LOẠI** | **LOGIC GATES** |
| PDF text: pdfplumber → PyPDF2 | Phát hiện tiếng Việt (>8%) → dịch sang tiếng Anh | Input: vector 384-dim từ BERT | AND: bắt buộc TẤT CẢ kỹ năng |
| PDF scan/JPG/PNG: tiền xử lý ảnh (grayscale, tăng tương phản, giảm nhiễu) → EasyOCR nhận dạng ký tự | Chuẩn hóa: lowercase, xóa URL/email, Skills Boost, Lemmatization | Hidden 1: 256n → BN → ReLU → Dropout(0.45) | OR: ít nhất 1 kỹ năng trong danh sách |
| | BERT Encoder → vector **384 chiều** mang ngữ cảnh đầy đủ | Hidden 2: 128n → BN → ReLU → Dropout(0.35) | NOT: loại nếu chứa từ khóa cấm |
| | | Output: 6 ngành → Softmax → **confidence%** | XOR: chọn đúng 1 trong 2 điều kiện |

**Kết quả cuối:** ĐẠT / LOẠI + confidence% + ngành phân loại

---

## SLIDE 6: CHAPTER HEADER — CHƯƠNG 2: PHÂN TÍCH ĐỀ TÀI
*(Slide nền đậm, chỉ có tiêu đề chương ở giữa)*

---

## SLIDE 7: PHÂN TÍCH YÊU CẦU
**Tiêu đề:** PHÂN TÍCH YÊU CẦU

- **Mục tiêu:** Xây dựng hệ thống tự động thay thế con người ở giai đoạn sơ loại CV — không keyword matching đơn giản mà dựa vào ngữ nghĩa thực sự
- **Chức năng chính:**
  - Đọc được nhiều định dạng CV (PDF có text, PDF scan, ảnh JPG/PNG)
  - Hiểu nội dung CV và phân loại ngành nghề phù hợp
  - Áp dụng điều kiện lọc theo quy tắc nghiệp vụ của HR (Logic Gates)
  - Hiển thị kết quả trực quan, lưu lịch sử sàng lọc
- **Thách thức kỹ thuật:**
  - CV có bố cục đa dạng (1 cột, 2 cột, sidebar, bảng) → khó trích xuất text
  - TF-IDF bag-of-words không phân biệt ngữ cảnh → nhầm ngành
  - Cần cân bằng giữa quy tắc cứng (Logic Gates) và dự đoán mềm (ML)
- **Định hướng giải quyết:** Pipeline 4 tầng BERT + MLP + Logic Gates


    ## SLIDE 8: YÊU CẦU CHỨC NĂNG
    **Tiêu đề:** YÊU CẦU CHỨC NĂNG

    *(5 khối flow ngang)*

    - **HR Setup** — Nhập tên vị trí tuyển dụng, chọn ngành, cấu hình kỹ năng yêu cầu và thiết lập Logic Gates (AND/OR/NOT/XOR)
    - **Input Management** — Upload PDF / JPG / PNG; hệ thống tự phát hiện định dạng (PDF text hay scan); kiểm tra file hợp lệ
    - **Core Processing** — Trích xuất text → phát hiện ngôn ngữ → dịch tiếng Việt → tiền xử lý → BERT encoding → MLP dự đoán → Logic Gates lọc
    - **Result Visualization** — Hiển thị bảng ứng viên với confidence%, ngành phân loại, trạng thái ĐẠT/LOẠI; lọc & sắp xếp; xuất CSV
    - **History Tracking** — Lưu lịch sử theo từng đợt tuyển dụng; tra cứu lại kết quả cũ; thống kê tỉ lệ đạt/loại

    ---

## SLIDE 9: CHAPTER HEADER — CHƯƠNG 3: THIẾT KẾ ỨNG DỤNG
*(Slide nền đậm, chỉ có tiêu đề chương ở giữa)*

---

## SLIDE 10: TẠI SAO CHỌN MLP?
**Tiêu đề:** LỰA CHỌN MÔ HÌNH: TẠI SAO MLP?

**Bài toán:** Phân loại CV vào 6 ngành dựa trên vector đặc trưng cố định — không cần xử lý chuỗi tuần tự, không cần nhớ ngữ cảnh dài

**Tại sao MLP phù hợp hơn các lựa chọn khác:**

| Lựa chọn | Lý do không dùng |
|---|---|
| SVM / Logistic Regression | Tốt với TF-IDF nhưng khó học đặc trưng phi tuyến phức tạp từ BERT embedding |
| CNN / RNN | Xử lý chuỗi tuần tự — không cần thiết khi đầu vào đã là vector tổng hợp 384-dim |
| Fine-tune toàn bộ BERT | Cần GPU mạnh, dữ liệu lớn — dataset ~1.197 mẫu quá nhỏ, dễ overfit |
| **MLP** | **Nhận vector cố định → học ánh xạ phi tuyến → ra xác suất — đúng với bài toán này** |

**Ưu điểm cụ thể của MLP trong hệ thống này:**
- Đầu vào là vector 384-dim đã mang đủ ngữ nghĩa từ BERT → MLP chỉ cần học cách phân loại, không cần hiểu văn bản
- Tham số ít (~132.000) — tránh overfit trên dataset ~1.197 mẫu
- Yêu cầu đề tài: áp dụng MLP — phần classification head là MLP thuần túy (BERT chỉ thay thế TF-IDF ở bước trích đặc trưng)

---

## SLIDE 11: TẠI SAO DÙNG BERT? (MỞ RỘNG)
**Tiêu đề:** MỞ RỘNG: BERT ENCODER — TẠI SAO THAY TF-IDF?

**Vấn đề của TF-IDF (phát hiện qua giai đoạn baseline):**
- TF-IDF chỉ đếm tần suất từ — không biết từ đó xuất hiện trong ngữ cảnh nào
- Hậu quả: MLP buộc phải ghi nhớ pattern bề mặt → **overfit 13.4%**

**Ví dụ điển hình dẫn đến quyết định thay thế:**
- Từ **"SEO"** trong CV Frontend Developer (TypeScript, React) → TF-IDF nhầm sang Digital Media
- Từ **"management"** có mặt ở CONSULTANT, BD, SALES → TF-IDF không phân biệt được

**BERT giải quyết như thế nào:**
- Không đếm từ mà hiểu toàn bộ câu → cùng từ, khác ngữ cảnh → vector khác nhau
- Chuyển toàn bộ CV thành **1 vector 384 chiều** chứa ngữ nghĩa đầy đủ → MLP chỉ việc phân loại

**Lý do chọn paraphrase-multilingual-MiniLM-L12-v2:**
- Hỗ trợ tiếng Việt — CV nộp vào có thể viết tiếng Việt
- Frozen weights (không train lại) — tiết kiệm tài nguyên, phù hợp phạm vi đồ án
- Output 384-dim — nhỏ gọn hơn BERT-base (768-dim), đủ chất lượng cho bài toán này

---

## SLIDE 12: TẦNG 3 — MLP CLASSIFICATION HEAD
**Tiêu đề:** TẦNG 3: MLP CLASSIFICATION HEAD

**Kiến trúc:**
```
Input Layer:    384 neuron  ← vector từ BERT Encoder
      ↓
Hidden Layer 1: 256 neuron → BatchNorm1d → ReLU → Dropout(0.45)
      ↓
Hidden Layer 2: 128 neuron → BatchNorm1d → ReLU → Dropout(0.35)
      ↓
Output Layer:     6 neuron → Softmax → confidence% cho mỗi ngành
```

- **BatchNorm1d:** Chuẩn hóa đầu ra mỗi lớp → ổn định quá trình training, hội tụ nhanh hơn
- **Dropout(0.45 / 0.35):** Ngẫu nhiên tắt neuron trong training → ngăn model ghi nhớ dữ liệu (overfitting)
- **Tổng tham số: ~132.000** — nhỏ gọn, phù hợp dataset ~1.197 mẫu
- **Optimizer:** AdamW (weight_decay=8e-3) — học nhanh, L2 regularization tích hợp
- *(Chèn Hình 3.1 Kiến trúc tổng quát MLP)*

---

## SLIDE 13: TẦNG 4 — LOGIC GATES (AND / OR)
**Tiêu đề:** TẦNG 4: LOGIC GATES — BỘ LỌC ĐIỀU KIỆN NGHIỆP VỤ

*(2 cột)*

**AND Gate — Bắt buộc TẤT CẢ điều kiện:**
- Ví dụ: Python **AND** SQL **AND** Docker
- Ứng viên chỉ ĐẠT nếu CV có đủ cả 3 kỹ năng
- Dùng cho: kỹ năng cốt lõi không thể thiếu
- Bảng chân trị:
  - A=0, B=0 → 0 | A=0, B=1 → 0
  - A=1, B=0 → 0 | A=1, B=1 → **1**

**OR Gate — Ít nhất 1 điều kiện:**
- Ví dụ: ReactJS **OR** VueJS **OR** Angular
- Ứng viên ĐẠT nếu biết ít nhất 1 framework frontend
- Dùng cho: nhóm kỹ năng thay thế nhau được
- Bảng chân trị:
  - A=0, B=0 → 0 | A=0, B=1 → **1**
  - A=1, B=0 → **1** | A=1, B=1 → **1**

---

## SLIDE 14: TẦNG 4 — LOGIC GATES (NOT / XOR)
**Tiêu đề:** TẦNG 4: LOGIC GATES (tiếp theo)

*(2 cột)*

**NOT Gate — Loại trừ điều kiện:**
- Ví dụ: **NOT** "fresher" | **NOT** "intern" | **NOT** "no experience"
- LOẠI ngay nếu CV chứa từ khóa cấm → tiết kiệm thời gian lọc thủ công
- Dùng cho: vị trí yêu cầu kinh nghiệm, loại ứng viên không đủ điều kiện cơ bản
- Bảng chân trị: NOT 0 → **1** | NOT 1 → **0**

**XOR Gate — Chọn đúng 1 trong 2:**
- Ví dụ: Frontend **XOR** Backend (chỉ tuyển chuyên sâu 1 hướng)
- ĐẠT nếu có đúng 1 kỹ năng, LOẠI nếu có cả hai hoặc không có cái nào
- Dùng cho: vị trí cần chuyên môn hóa, tránh ứng viên "biết hết nhưng không sâu"
- Bảng chân trị:
  - A=0, B=0 → 0 | A=0, B=1 → **1**
  - A=1, B=0 → **1** | A=1, B=1 → **0**

> **Lưu ý:** Logic Gates là advisory — nếu LOẠI nhưng confidence ≥ 30% → hệ thống cảnh báo HR xem lại thủ công

---

## SLIDE 15: QUY TRÌNH XỬ LÝ TỔNG THỂ
**Tiêu đề:** QUY TRÌNH XỬ LÝ TỔNG THỂ

*(Slide diagram flowchart — chèn Hình 3.2 từ báo cáo)*

**Sơ đồ các bước:**
```
Upload CV (PDF / JPG / PNG)
        ↓
Tầng 1 — Trích xuất text:
  PDF text  → pdfplumber → PyPDF2 (dự phòng nếu < 50 ký tự)
  PDF scan / Ảnh → Tiền xử lý ảnh → EasyOCR
        ↓
Phát hiện ngôn ngữ:
  Tỉ lệ ký tự tiếng Việt > 8% → dịch sang tiếng Anh (GoogleTranslator)
        ↓
Tầng 2 — Tiền xử lý + BERT:
  Lowercase → xóa URL/email → chuẩn hóa viết tắt → Skills Boost
  → Lemmatization → BERT Encoder → vector 384-dim
        ↓
Tầng 3 — MLP Head:
  384 → 256 → 128 → 6 ngành → Softmax → confidence%
        ↓
Tầng 4 — Logic Gates:
  AND / OR / NOT / XOR → ĐẠT / LOẠI
  (confidence < 30% + LOẠI → cảnh báo HR)
        ↓
Kết quả: ĐẠT/LOẠI + confidence% + ngành phân loại + điểm logic
```

---

## SLIDE 16: PHƯƠNG PHÁP TRÍCH XUẤT TEXT
**Tiêu đề:** PHƯƠNG PHÁP TRÍCH XUẤT VĂN BẢN

**3 lớp xử lý theo thứ tự ưu tiên:**

- **Lớp 1 — pdfplumber (ưu tiên):**
  - Dùng `extract_words()` lấy tọa độ từng từ trên trang
  - Nhóm các từ cùng vị trí dòng (top, làm tròn 5px) → đọc đúng thứ tự trên-xuống-dưới, trái-sang-phải
  - Xử lý tốt CV đa cột, sidebar dọc, bảng kỹ năng dạng lưới
  - Nếu trích xuất < 50 ký tự → chuyển sang Lớp 2

- **Lớp 2 — PyPDF2 (dự phòng):**
  - Đọc text layer trực tiếp theo từng trang, ghép nội dung lại
  - Phù hợp CV đơn giản 1 cột tạo bằng Word / Google Docs
  - Nếu vẫn < 50 ký tự → chuyển sang Lớp 3

- **Lớp 3 — EasyOCR (fallback):**
  - Áp dụng cho PDF scan / JPG / PNG (không có text layer)
  - Tiền xử lý ảnh: chuyển grayscale, tăng tương phản, giảm nhiễu, resize
  - Nhận dạng ký tự → xuất văn bản thô

**Bảng so sánh:**

| Tiêu chí | pdfplumber | PyPDF2 | EasyOCR |
|---|---|---|---|
| CV đa cột / sidebar | Tốt (sắp xếp tọa độ) | Dễ lẫn lộn cột | Không áp dụng |
| CV đơn giản 1 cột | Tốt | Tốt | Không áp dụng |
| PDF scan / ảnh | Không hỗ trợ | Không hỗ trợ | Hỗ trợ (OCR) |
| Tốc độ xử lý | Trung bình | Nhanh | Chậm |

---

## SLIDE 17: PHƯƠNG PHÁP TIỀN XỬ LÝ DỮ LIỆU
**Tiêu đề:** PHƯƠNG PHÁP TIỀN XỬ LÝ DỮ LIỆU

**Các bước thực hiện theo thứ tự:**

1. **Phát hiện tiếng Việt:** Đếm ký tự Unicode đặc trưng (ă, ơ, ư, đ...) — nếu > 8% tổng ký tự alpha → dịch sang tiếng Anh (GoogleTranslator, chia chunk 4.500 ký tự)
   - Ngưỡng 8%: tránh dịch nhầm CV tiếng Anh chỉ có tên/địa chỉ tiếng Việt
2. **Skills Boosting:** Tìm phần "Skills" trong CV → nhân đôi nội dung → từ kỹ năng có trọng số cao hơn khi BERT encoding
3. **Lowercase:** Chuyển toàn bộ về chữ thường (Python = PYTHON = python)
4. **Xóa nhiễu:** Loại bỏ URL, địa chỉ email, số điện thoại, ký tự đặc biệt
5. **Chuẩn hóa viết tắt:** ML → machine learning | AI → artificial intelligence | NLP → natural language processing | JS → javascript
6. **Lemmatization:** Đưa từ về dạng gốc (running → run, experiences → experience, managed → manage)
7. **Loại stop words:** Xóa từ không mang nghĩa (the, a, is, are, in...) bằng NLTK

---

## SLIDE 18: NGUỒN GỐC VÀ THÔNG SỐ DATASET
**Tiêu đề:** NGUỒN GỐC VÀ THÔNG SỐ DATASET

**3 nguồn dữ liệu kết hợp:**

- **Nguồn 1 — Kaggle UpdatedResumeDataSet:**
  - 2.484 hồ sơ CV thực tế tiếng Anh, 24 ngành nghề
  - Lọc lấy 6 ngành mục tiêu → **678 mẫu** làm nền tảng
  - Nguồn uy tín, dữ liệu thực từ người dùng nộp CV

- **Nguồn 2 — Neuralframe Dataset:**
  - 9.544 mẫu có cấu trúc chi tiết (career_objective, skills, experience, education...)
  - Bổ sung tối đa **80 mẫu/ngành** để tránh mất cân bằng
  - Đánh dấu tiền tố ID "NF_"

- **Nguồn 3 — Synthetic Data (tự sinh):**
  - Module template-based generation: mỗi ngành có keyword pool riêng (kỹ năng, công cụ, chứng chỉ, vai trò)
  - Chọn ngẫu nhiên và ghép thành CV hoàn chỉnh → **235 mẫu** bổ sung cho ngành còn thiếu
  - Đánh dấu tiền tố "SYN_"

**6 ngành phân loại:** INFORMATION-TECHNOLOGY | BUSINESS-DEVELOPMENT | SALES | DIGITAL-MEDIA | PUBLIC-RELATIONS | CONSULTANT

---

## SLIDE 19: CẤU TRÚC VÀ PHÂN BỔ DỮ LIỆU
**Tiêu đề:** CẤU TRÚC VÀ PHÂN BỔ DỮ LIỆU

| Ngành | Kaggle | Neuralframe | Synthetic | Tổng |
|---|---|---|---|---|
| INFORMATION-TECHNOLOGY | 120 | 80 | 0 | 200 |
| BUSINESS-DEVELOPMENT | 120 | 40 | 41 | 201 |
| SALES | 116 | 56 | 28 | 200 |
| PUBLIC-RELATIONS | 111 | 80 | 5 | 196 |
| CONSULTANT | 115 | 28 | 57 | 200 |
| DIGITAL-MEDIA | 96 | 0 | 104 | 200 |
| **Tổng** | **678** | **284** | **235** | **~1.197** |

- **Tỉ lệ max/min = 1.03x** — gần như cân bằng tuyệt đối giữa 6 ngành
- Mô hình học đều tất cả ngành, không bị thiên kiến về ngành nào
- DIGITAL-MEDIA cần nhiều Synthetic (104) vì Neuralframe không có dữ liệu ngành này
- CONSULTANT cần nhiều Synthetic (57) vì dữ liệu gốc từ Kaggle ít đặc trưng riêng biệt

---

## SLIDE 20: PHÂN CHIA DỮ LIỆU — TRAIN / VAL / TEST
**Tiêu đề:** PHÂN CHIA DỮ LIỆU

| Tập | Tỉ lệ | Số mẫu | Mục đích |
|---|---|---|---|
| Train | 70% | ~838 mẫu | Huấn luyện trọng số model |
| Validation | 15% | ~180 mẫu | Early Stopping — chọn checkpoint tốt nhất theo Val Macro F1 |
| Test | 15% | ~180 mẫu | Đánh giá cuối cùng — **chỉ dùng 1 lần duy nhất** (unbiased) |

- **Phương pháp:** Stratified Split — mỗi tập đều có tỉ lệ 6 ngành như nhau
- **Đồng bộ giữa MLP và BERT:** Cùng random seed 42, cùng tỉ lệ → so sánh công bằng (apples-to-apples)
- **Tập Test độc lập hoàn toàn:** Không dùng trong huấn luyện lẫn chọn hyperparameter → kết quả phản ánh đúng khả năng tổng quát hóa

---

## SLIDE 21: CHAPTER HEADER — CHƯƠNG 4: THỰC HIỆN CÀI ĐẶT
*(Slide nền đậm, chỉ có tiêu đề chương ở giữa)*

---

## SLIDE 22: CÔNG CỤ VÀ MÔI TRƯỜNG PHÁT TRIỂN
**Tiêu đề:** CÔNG CỤ VÀ MÔI TRƯỜNG PHÁT TRIỂN

*(2 cột)*

**Deep Learning & NLP:**
- **PyTorch ≥ 2.0** — framework training model
- **Sentence-Transformers ≥ 2.2** — BERT encoder (frozen)
- **Scikit-learn ≥ 1.2** — chia tập dữ liệu, đánh giá (F1, accuracy, confusion matrix)
- **NLTK** — Lemmatization, danh sách stop words tiếng Anh

**PDF / OCR / Dịch thuật:**
- **pdfplumber ≥ 0.10** — trích xuất text PDF theo tọa độ
- **PyPDF2** — đọc text layer PDF đơn giản (fallback)
- **EasyOCR ≥ 1.7** — nhận dạng ký tự từ ảnh scan
- **opencv-python** — tiền xử lý ảnh cho OCR
- **deep-translator** — dịch tiếng Việt → tiếng Anh (GoogleTranslator)
- **langdetect** — nhận diện ngôn ngữ văn bản

**Web Backend:**
- **FastAPI ≥ 0.110** — REST API backend
- **Uvicorn** — ASGI server chạy FastAPI
- **Jinja2** — render HTML templates server-side

---

## SLIDE 23: GIAI ĐOẠN 1 — MLP BASELINE (TF-IDF)
**Tiêu đề:** GIAI ĐOẠN 1: THỬ NGHIỆM MLP BASELINE

**Mục tiêu:** Xác lập đường cơ sở (baseline) và phát hiện giới hạn cần khắc phục

**Kiến trúc:**
```
TF-IDF (600 đặc trưng) → Linear(64) → BatchNorm → ReLU → Dropout(0.4)
                       → Linear(32) → BatchNorm → ReLU → Dropout(0.3)
                       → Linear(6)  → Softmax
Tổng tham số: ~41.000
```

**Siêu tham số:**

| Tham số | Giá trị | Lý do chọn |
|---|---|---|
| Epochs tối đa | 150 | Đủ để hội tụ |
| Batch size | 64 | Cân bằng tốc độ và ổn định |
| Learning rate | 0.001 | LR tiêu chuẩn cho Adam |
| Scheduler | CosineAnnealingLR | Giảm LR mượt theo chu kỳ cosine |
| Early stop | 8 epoch | Dừng sớm tránh overfit |
| Label smoothing | 0.1 | Giảm overconfidence |
| Gradient clipping | 1.0 | Ổn định training |

**Kết quả:**
- Best Val F1: **87.73%** | Val Accuracy: **87.71%** | Epoch thực tế: **17**
- Test Accuracy: **85.56%** | Test Macro F1: **85.67%**
- **Overfit gap: ~13.4%** ⚠ (Train ~99% vs Test 85.56%)

---

## SLIDE 24: NGUYÊN NHÂN OVERFIT — TF-IDF
**Tiêu đề:** NGUYÊN NHÂN OVERFIT — TF-IDF BAG-OF-WORDS

**Nguyên nhân cốt lõi:** TF-IDF chỉ đếm tần suất từ — KHÔNG phân biệt ngữ cảnh

**Ví dụ điển hình:**
- Từ **"SEO"** trong CV Frontend Developer (có TypeScript, Next.js, React)
- Từ **"SEO"** trong CV Digital Marketing Specialist
- TF-IDF: hai CV này có vector giống nhau tại chiều "SEO" → **nhầm ngành**
- BERT: hiểu rằng "SEO" trong ngữ cảnh TypeScript/React là của IT developer → **phân loại đúng**

**Các trường hợp dễ nhầm với TF-IDF:**
- "management" xuất hiện ở CONSULTANT, BUSINESS-DEVELOPMENT, SALES → ranh giới mờ
- "communication" có ở PUBLIC-RELATIONS lẫn SALES → khó phân biệt
- MLP buộc phải **ghi nhớ pattern bề mặt** thay vì học đặc trưng thực sự → overfit

**→ Giải pháp:** Thay TF-IDF bằng BERT Contextual Embedding — cùng từ nhưng ngữ cảnh khác → vector khác nhau

---

## SLIDE 25: GIAI ĐOẠN 2 — BERT + MLP (PIPELINE CHÍNH THỨC)
**Tiêu đề:** GIAI ĐOẠN 2: BERT + MLP HEAD

**Thiết kế:**
```
paraphrase-multilingual-MiniLM-L12-v2 (FROZEN)
        ↓ vector 384-dim (ngữ nghĩa đầy đủ)
Linear(384→256) → BatchNorm1d → ReLU → Dropout(0.45)
        ↓
Linear(256→128) → BatchNorm1d → ReLU → Dropout(0.35)
        ↓
Linear(128→6) → Softmax → confidence%
```

> Classification head vẫn là **MLP thuần túy** → đúng yêu cầu đề tài "Áp dụng MLP"
> BERT encoder chỉ thay thế TF-IDF ở bước trích xuất đặc trưng

**Siêu tham số:**

| Tham số | Giá trị | Lý do chọn |
|---|---|---|
| Epochs tối đa | 80 | BERT embedding chất lượng cao → hội tụ nhanh hơn |
| Batch size | 32 | Nhỏ hơn MLP → gradient ổn định hơn |
| Learning rate | 2e-4 | Thấp hơn MLP vì embedding đã tốt |
| Optimizer | AdamW | Adam + L2 regularization tích hợp |
| Weight decay | 8e-3 | Regularization mạnh hơn |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) | Chỉ giảm LR khi plateau thực sự |
| Early stopping | 15 epoch | Patience lớn hơn do hội tụ chậm hơn |

> **Tại sao ReduceLROnPlateau thay CosineAnnealingLR?**
> CosineAnnealing gây dao động Val F1 ±3% — ReduceLROnPlateau chỉ giảm LR khi thực sự cần → hội tụ mượt và ổn định hơn

---

## SLIDE 26: ĐÁNH GIÁ TRAINING — MLP BASELINE
**Tiêu đề:** ĐÁNH GIÁ HUẤN LUYỆN — MLP BASELINE

*(Chèn hình: `Models/training_curves.png`)*

**Nhận xét từ đồ thị (3 biểu đồ: Loss / Accuracy / Val F1 theo epoch):**
- Model hội tụ rất nhanh, dừng tại epoch 17 (early stopping)
- **Train Accuracy tăng lên ~99%** trong khi **Val Accuracy chỉ đạt ~87%**
- Khoảng cách lớn giữa Train và Val → dấu hiệu **overfitting rõ rệt (~13.4%)**
- Val F1 đạt đỉnh sớm rồi không cải thiện thêm → model đã học thuộc tập train

---

## SLIDE 27: ĐÁNH GIÁ TRAINING — BERT + MLP
**Tiêu đề:** ĐÁNH GIÁ HUẤN LUYỆN — BERT + MLP

*(Chèn hình: `Models/bert_training_curves.png`)*

**Nhận xét từ đồ thị:**
- Model hội tụ trong ~42 epoch
- **Train Accuracy ≈ Val Accuracy** xuyên suốt quá trình — không có dấu hiệu overfitting
- Val F1 tăng ổn định và đạt đỉnh **91.54%** trước khi dừng
- **Overfit gap chỉ ~2.1%** (Train ~91% vs Test 88.89%)
- Bằng chứng: nguyên nhân overfit của MLP Baseline là đặc trưng TF-IDF, **không phải kiến trúc MLP**

---

## SLIDE 28: ĐÁNH GIÁ VÀ SO SÁNH HAI MÔ HÌNH
**Tiêu đề:** ĐÁNH GIÁ VÀ SO SÁNH HAI MÔ HÌNH

| Chỉ số | MLP Baseline (TF-IDF) | BERT + MLP ✓ | Cải thiện |
|---|---|---|---|
| Val Accuracy | 87.71% | **91.62%** | +3.91% |
| Val Macro F1 | 87.73% | **91.54%** | +3.81% |
| Test Accuracy | 85.56% | **88.89%** | +3.33% |
| Test Macro F1 | 85.67% | **88.84%** | +3.17% |
| Overfit gap | ~13.4% ⚠ | **~2.1% ✓** | Giảm 11.3% |
| Embedding | Bag-of-words (600 dim) | Contextual (384 dim) | — |
| Tham số | ~41.000 | ~132.000 | — |
| Epoch thực tế | 17 | ~42 | — |

**Kết luận:** Cùng kiến trúc MLP — thay TF-IDF bằng BERT → Test Accuracy tăng 85.56% → 88.89%, overfit giảm từ 13.4% xuống 2.1%

---

## SLIDE 29: CONFUSION MATRIX — BERT + MLP
**Tiêu đề:** CONFUSION MATRIX — BERT + MLP

*(Chèn hình: `Models/confusion_matrix.png`)*

---

## SLIDE 30: PHÂN TÍCH CONFUSION MATRIX
**Tiêu đề:** PHÂN TÍCH KẾT QUẢ

- **Đường chéo đậm** → mô hình phân loại chính xác đa số ngành
- **INFORMATION-TECHNOLOGY:** ít nhầm nhất — từ vựng kỹ thuật (Python, API, Docker...) riêng biệt, BERT dễ phân biệt
- **CONSULTANT ↔ BUSINESS-DEVELOPMENT:** hay nhầm nhau — CV tư vấn đề cập nhiều lĩnh vực (IT, kinh doanh, quản lý) → ranh giới mờ, ~200 mẫu/ngành còn nhỏ
- **PUBLIC-RELATIONS ↔ DIGITAL-MEDIA:** nhầm do cả hai liên quan truyền thông, nội dung, thương hiệu — từ vựng giao thoa cao
- **SALES ↔ BUSINESS-DEVELOPMENT:** nhầm do từ vựng doanh thu, khách hàng, mục tiêu tương đồng
- **Không có nhầm lẫn chéo xa:** IT không nhầm sang SALES, BD không nhầm sang CONSULTANT → model nắm được đặc trưng tổng thể từng nhóm ngành

---

## SLIDE 31: ĐÁNH GIÁ TRÊN DỮ LIỆU THỰC TẾ (60 CV)
**Tiêu đề:** ĐÁNH GIÁ NGOÀI TẬP HUẤN LUYỆN — OUT-OF-SAMPLE

*(Chèn hình đánh giá — Hình 4.3 từ báo cáo)*

**60 CV thực tế thu thập độc lập (10 CV × 6 ngành), nằm ngoài tập train:**

| Ngành | Kết quả | Accuracy | Nhận xét |
|---|---|---|---|
| BUSINESS-DEVELOPMENT | 10/10 | **100%** | Từ vựng đặc trưng rõ ràng |
| INFORMATION-TECHNOLOGY | 9/10 | **90%** | Rất tốt, 1 CV nhầm sang CONSULTANT |
| DIGITAL-MEDIA | 9/10 | **90%** | Tốt, 1 CV nhầm sang PUBLIC-RELATIONS |
| PUBLIC-RELATIONS | 7/10 | 70% | Nhầm 3 CV sang DIGITAL-MEDIA |
| SALES | 6/10 | 60% | Nhầm 4 CV sang BUSINESS-DEVELOPMENT |
| CONSULTANT | 5/10 | 50% | Thấp nhất — CV đa lĩnh vực, ranh giới mờ |
| **Tổng** | **46/60** | **76.7%** | Cao hơn ngẫu nhiên 16.7% → model tổng quát tốt |

---

## SLIDE 32: PHÂN TÍCH ĐỘ TIN CẬY (CONFIDENCE)
**Tiêu đề:** PHÂN TÍCH ĐỘ TIN CẬY (CONFIDENCE)

*(2 cột so sánh)*

**Dự đoán ĐÚNG (46 CV — màu xanh):**
- Confidence tập trung **70–100%**, đỉnh tại 80–90%
- Mô hình đưa ra quyết định với mức chắc chắn cao
- **Không đoán mò** — có cơ sở ngữ nghĩa rõ ràng từ BERT embedding

**Dự đoán SAI (14 CV — màu đỏ):**
- Confidence tập trung **30–60%**, đỉnh tại 35–45%
- Mô hình "nhận thức được" sự không chắc chắn của mình
- Khi sai, model thường cho confidence thấp → có thể dùng ngưỡng để lọc

**Ý nghĩa thực tế:**
- Có thể đặt **ngưỡng confidence** để phân loại rủi ro: < 50% → cần HR xem lại thủ công
- Tầng 4 Logic Gates: nếu kết quả LOẠI nhưng confidence ≥ 30% → hiển thị cảnh báo cho HR

---

## SLIDE 33: GIAO DIỆN DASHBOARD
**Tiêu đề:** GIAO DIỆN DASHBOARD

*(Chèn screenshot Hình 4.4)*

**Chức năng hiển thị:**
- Tổng số CV đã xử lý | Số CV đạt / loại | Tỉ lệ đạt
- Biểu đồ phân bổ CV theo ngành nghề
- Danh sách hoạt động gần nhất (file CV + trạng thái)
- Thông tin job đang tuyển dụng

---

## SLIDE 34: GIAO DIỆN THIẾT LẬP JOB
**Tiêu đề:** GIAO DIỆN THIẾT LẬP JOB

*(Chèn screenshot Hình 4.5)*

**Chức năng:**
- Chọn vị trí tuyển dụng (6 ngành có sẵn)
- Cấu hình AND Gate: nhập danh sách kỹ năng bắt buộc
- Cấu hình OR Gate: nhập nhóm kỹ năng ưu tiên
- Cấu hình NOT Gate: nhập từ khóa cấm
- Preview tóm tắt điều kiện Logic Gates đã thiết lập

---

## SLIDE 35: GIAO DIỆN UPLOAD CV
**Tiêu đề:** GIAO DIỆN UPLOAD CV

*(Chèn screenshot Hình 4.6)*

**Chức năng:**
- Kéo thả hoặc chọn file (PDF / JPG / PNG)
- Upload nhiều CV cùng lúc
- Xử lý real-time: hiển thị trạng thái từng file
- Bảng xếp hạng ứng viên cập nhật ngay sau khi xử lý xong

---

## SLIDE 36: GIAO DIỆN KẾT QUẢ SÀNG LỌC
**Tiêu đề:** GIAO DIỆN KẾT QUẢ SÀNG LỌC

*(Chèn screenshot Hình 4.7)*

**Chức năng:**
- Bảng tổng hợp ứng viên: tên file, ngành phân loại, confidence%, ngôn ngữ CV, trạng thái ĐẠT/LOẠI
- Lọc theo trạng thái (Đạt / Loại / Tất cả) và theo ngành
- Sắp xếp theo confidence% giảm dần
- Xuất danh sách ra file CSV
- Click vào từng dòng → xem chi tiết CV

---

## SLIDE 37: CHI TIẾT CV ĐƯỢC ĐÁNH GIÁ
**Tiêu đề:** CHI TIẾT CV ĐƯỢC ĐÁNH GIÁ

*(Chèn screenshot Hình 4.10)*

**Thông tin hiển thị:**
- Nội dung text CV đã trích xuất và tiền xử lý
- Ngành phân loại + confidence% + biểu đồ xác suất 6 ngành
- Kết quả từng Logic Gate (AND ✓/✗ | OR ✓/✗ | NOT ✓/✗ | XOR ✓/✗)
- Lý do kết quả cuối: ĐẠT / LOẠI
- Cảnh báo nếu LOẠI nhưng confidence ≥ 30%

---

## SLIDE 38: CHAPTER HEADER — CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
*(Slide nền đậm, chỉ có tiêu đề chương ở giữa)*

---

## SLIDE 39: KẾT LUẬN
**Tiêu đề:** KẾT LUẬN

- **Về mô hình:** Huấn luyện thành công 2 mô hình trên dataset ~1.197 mẫu cân bằng (1.03x)
  - MLP Baseline (TF-IDF): Val F1 = 87.73%, Test Acc = 85.56%, overfit **~13.4%**
  - BERT + MLP: Val F1 = 91.54%, Test Acc = **88.89%**, overfit chỉ **~2.1%**
- **Về dữ liệu:** Chủ động mở rộng từ 678 → 1.197 mẫu bằng 3 nguồn (Kaggle + Neuralframe + Synthetic)
- **Về ứng dụng:** Web app FastAPI hoàn chỉnh — hỗ trợ PDF, ảnh scan, tiếng Việt, logic gates, lịch sử, xuất CSV
- **Kết luận quan trọng nhất:**
  - Nguyên nhân overfit **KHÔNG phải** kiến trúc MLP
  - Nguyên nhân là **chất lượng đặc trưng đầu vào (TF-IDF)**
  - Khi thay TF-IDF bằng BERT: cùng một kiến trúc MLP → Test Acc **85.56% → 88.89%**, overfit **13.4% → 2.1%**

---

## SLIDE 40: HƯỚNG PHÁT TRIỂN
**Tiêu đề:** HƯỚNG PHÁT TRIỂN

- **Mở rộng dataset & ngành:** Thu thập CV thực tế Việt Nam; thêm ngành Healthcare, Education, Finance, Engineering; áp dụng augmentation (back-translation, synonym replacement)
- **Cải thiện mô hình:** Fine-tuning toàn bộ BERT encoder (hiện đang frozen) → kỳ vọng tăng thêm 2–5% F1; nghiên cứu các BERT variant khác (PhoBERT cho tiếng Việt)
- **Explainable AI:** Tích hợp SHAP hoặc LIME → cung cấp lý do cụ thể cho từng quyết định Đạt/Loại → tăng tính minh bạch và tin cậy với HR
- **Phân tích đa tiêu chí:** Bổ sung số năm kinh nghiệm, bằng cấp, chứng chỉ (không chỉ phân loại ngành) → đánh giá toàn diện hơn
- **Triển khai production:** Docker + AWS/GCP → phục vụ nhiều người dùng đồng thời với khả năng mở rộng linh hoạt; tích hợp API với các hệ thống ATS (Applicant Tracking System) phổ biến

---

## SLIDE 41: THANK YOU
- **THANK YOU!**
- Nhóm 14 — Đồ án Deep Learning — HUIT
- GVHD: TS. Nguyễn Thanh Long
