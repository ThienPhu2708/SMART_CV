import torch
import numpy as np
import os
import pickle
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from SRC.Processing.extractor import extract_text
from SRC.Processing.cleaner import clean_text, translate_if_needed
from SRC.model_mlp import SmartCV_MLP
from SRC.logic_gates import check_mandatory_skills


def run_prediction(pdf_path, required_skills):
    """
    Pipeline: PDF -> OCR -> Cleaning -> Logic Filter -> AI Classification
    Returns dict with result or None on failure.
    """
    print(f"\n--- ĐANG PHÂN TÍCH HỒ SƠ: {os.path.basename(pdf_path)} ---")

    # BƯỚC 1: Trích xuất văn bản
    print("Bước 1: Đang trích xuất văn bản từ PDF...")
    try:
        raw_text = extract_text(pdf_path)
    except Exception as e:
        print(f"Lỗi trích xuất: {e}")
        return None

    # BƯỚC 2: Dịch sang tiếng Anh nếu CV không phải tiếng Anh
    print("Bước 2: Kiểm tra và dịch ngôn ngữ (nếu cần)...")
    raw_text = translate_if_needed(raw_text)

    # BƯỚC 3: Làm sạch văn bản
    print("Bước 3: Đang làm sạch dữ liệu...")
    cleaned_text = clean_text(raw_text)

    # BƯỚC 4: Logic Gate - kiểm tra kỹ năng bắt buộc (chỉ cảnh báo, không chặn MLP)
    print("Bước 4: Kiểm tra kỹ năng bắt buộc (Logic AND Gate)...")
    is_qualified = check_mandatory_skills(cleaned_text, required_skills)

    # BƯỚC 5: Dự đoán bằng MLP — luôn chạy bất kể Logic Gate
    print("Bước 5: Đang phân loại bằng mạng Neural MLP...")
    try:
        if not os.path.exists("Models/classes.npy") or not os.path.exists("Models/smartcv_model.pth"):
            print("Lỗi: Thiếu file model. Hãy chạy train.py trước!")
            return None

        classes = np.load("Models/classes.npy", allow_pickle=True)

        vectorizer_path = "Data/Processed/tfidf_vectorizer.pkl"
        if not os.path.exists(vectorizer_path):
            print("Lỗi: Thiếu file vectorizer. Hãy chạy main.py trước!")
            return None

        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)

        input_vector = vectorizer.transform([cleaned_text]).toarray()
        input_size = input_vector.shape[1]          # dynamic, không hardcode
        input_tensor = torch.tensor(input_vector).float()

        model = SmartCV_MLP(input_size=input_size, num_classes=len(classes))
        model.load_state_dict(torch.load("Models/smartcv_model.pth", map_location="cpu"))
        model.eval()

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            category = classes[predicted.item()]
            confidence_pct = confidence.item() * 100

        all_probs = {classes[i]: probabilities[0][i].item() * 100 for i in range(len(classes))}

        # Kết hợp: Logic Gate quyết định status, MLP luôn trả kết quả
        if is_qualified:
            status = "ĐẠT"
            reason = "Đáp ứng tất cả tiêu chí"
        else:
            missing = [s for s in required_skills
                       if not check_mandatory_skills(cleaned_text, [s])]
            status = "LOẠI"
            reason = f"Thiếu kỹ năng bắt buộc: {', '.join(missing)}"

        print(f"KẾT QUẢ: {status}")
        print(f"Chuyên môn dự đoán: {category}")
        print(f"Độ tin cậy: {confidence_pct:.2f}%")
        if status == "LOẠI":
            print(f"Lưu ý: MLP vẫn dự đoán ngành → HR nên xem xét lại tiêu chí")

        return {
            "status":     status,
            "reason":     reason,
            "category":   category,
            "confidence": confidence_pct,
            "text":       cleaned_text,
            "all_probs":  all_probs,
        }

    except Exception as e:
        print(f"Lỗi hệ thống AI: {e}")
        return None


if __name__ == "__main__":
    REQUIRED_SKILLS = []
    CV_PATH = "Data/Raw/cv_scan.pdf"

    if os.path.exists(CV_PATH):
        run_prediction(CV_PATH, REQUIRED_SKILLS)
    else:
        print(f"Không tìm thấy file: {CV_PATH}")
