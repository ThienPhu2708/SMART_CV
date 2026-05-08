🚀 DỰ ÁN SMARTCV: HỆ THỐNG SÀNG LỌC HỒ SƠ TỰ ĐỘNG
📝 GIỚI THIỆU TỔNG QUAN
SmartCV là giải pháp đột phá trong lĩnh vực tuyển dụng, sử dụng trí tuệ nhân tạo để tối ưu hóa quy trình sơ loại hồ sơ ứng viên. Hệ thống kết hợp sức mạnh của học sâu và tư duy logic nghiệp vụ để đưa ra những đánh giá chính xác, khách quan nhất.

Đặc điểm nổi bật:
Deep Learning: Sử dụng mạng nơ-ron MLP (Multi-Layer Perceptron) để phân tích các đặc trưng phi tuyến tính từ kỹ năng và kinh nghiệm ứng viên.

Logic Gates: Tích hợp bộ lọc logic AND, OR, NOT, XOR giúp tùy biến các tiêu chí tuyển dụng "cứng" theo nhu cầu thực tế của doanh nghiệp.

📁 CẤU TRÚC THƯ MỤC DỰ ÁN
Dự án được tổ chức khoa học để phục vụ quá trình phát triển nhóm và vận hành hệ thống:

📂 Data/: Quản lý dữ liệu đầu vào.

- Raw/: Chứa Dataset gốc (Resume.csv) và các tệp CV mẫu định dạng PDF.

- Processed/: Lưu trữ dữ liệu sau khi đã được chuẩn hóa và chuyển đổi số.

📂 Models/: Lưu trữ các phiên bản mô hình MLP đã huấn luyện thành công.

📂 Notebooks/: Các tệp Jupyter Notebook dùng để thử nghiệm và phân tích dữ liệu.

📂 SRC/: Mã nguồn lõi của ứng dụng.

- Processing/: Module trích xuất văn bản (PDF) và nhận diện ký tự (OCR).

- model_mlp.py: Định nghĩa kiến trúc mạng nơ-ron nhân tạo.-

- logic_gates.py: Cài đặt các quy tắc logic nghiệp vụ.

📂 App/: Giao diện người dùng và Dashboard điều khiển.

📄 main.py: Tệp thực thi chính, kết nối toàn bộ các module của hệ thống.

🛠 CÔNG NGHỆ VÀ THƯ VIỆN SỬ DỤNG
Ngôn ngữ chính: Python 3.x

Xử lý dữ liệu: Pandas, Numpy, Scikit-learn (TF-IDF)

Học sâu: PyTorch hoặc TensorFlow

Thị giác máy tính & OCR: OpenCV, EasyOCR, PyPDF2
