"""
predict.py — Pipeline 4 tầng: OCR → BERT Encoder → MLP Head → Logic Gates
"""
import os
import torch

from SRC.Processing.extractor import extract_text
from SRC.Processing.cleaner import clean_text, translate_if_needed
from SRC.Processing.bert_encoder import encode_single
from SRC.model_bert import SmartCV_BERT
from SRC.logic_gates import evaluate_candidate, check_keyword

BERT_MODEL_PATH  = "Models/bert_classifier.pth"
BERT_CONFIG_PATH = "Models/bert_config.json"


def load_model():
    import json
    if not (os.path.exists(BERT_MODEL_PATH) and os.path.exists(BERT_CONFIG_PATH)):
        raise FileNotFoundError("Chưa có model. Chạy train_bert.py trước!")
    with open(BERT_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    model = SmartCV_BERT(embed_dim=config["embed_dim"], num_classes=config["num_classes"])
    model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, config


def run_prediction(cv_path: str, required_skills: list = None, optional_skills: list = None,
                   blacklist: list = None, exclusive_pair: list = None):
    """
    Pipeline: CV file → Text → Translate → Clean → BERT → MLP → Logic Gates
    Trả về dict kết quả hoặc None nếu lỗi.
    """
    required_skills  = required_skills or []
    optional_skills  = optional_skills or []
    blacklist        = blacklist or []
    exclusive_pair   = exclusive_pair or []

    print(f"\n--- ĐANG PHÂN TÍCH: {os.path.basename(cv_path)} ---")

    # Tầng 1: Trích xuất văn bản
    print("Tầng 1: Trích xuất văn bản (PDF/OCR)...")
    try:
        raw_text = extract_text(cv_path)
    except Exception as e:
        print(f"  Lỗi trích xuất: {e}")
        return None
    if not raw_text or len(raw_text.strip()) < 30:
        print("  Không trích xuất được nội dung.")
        return None

    # Dịch nếu không phải tiếng Anh
    raw_text = translate_if_needed(raw_text)
    cleaned  = clean_text(raw_text)

    # Tầng 2: BERT Encoder → vector 384 chiều
    print("Tầng 2: BERT Encoder mã hóa ngữ nghĩa...")
    try:
        model, config = load_model()
    except FileNotFoundError as e:
        print(f"  {e}")
        return None

    embedding = encode_single(cleaned)
    tensor    = torch.tensor(embedding, dtype=torch.float32)

    # Tầng 3: MLP Head phân loại ngành
    print("Tầng 3: MLP Head phân loại ngành...")
    classes = config["classes"]
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(tensor), dim=1)[0]
    conf, pred = torch.max(probs, 0)
    category   = classes[pred.item()]
    confidence = float(conf) * 100
    all_probs  = {classes[i]: float(probs[i]) * 100 for i in range(len(classes))}

    # Tầng 4: Logic Gates
    print("Tầng 4: Logic Gates (AND/OR/NOT/XOR)...")
    logic_result = evaluate_candidate(
        cleaned,
        required_skills=required_skills,
        optional_skills=optional_skills,
        blacklist=blacklist,
        exclusive_pair=exclusive_pair if len(exclusive_pair) == 2 else None,
    )

    if logic_result["overall_pass"]:
        status = "ĐẠT"
        reason = "Đáp ứng tất cả tiêu chí"
    else:
        reasons = []
        if not logic_result.get("mandatory_pass"):
            missing = [s for s in required_skills if not check_keyword(cleaned, s)]
            if missing:
                reasons.append(f"Thiếu kỹ năng: {', '.join(missing)}")
        if not logic_result.get("blacklist_pass"):
            reasons.append("Chứa từ khóa bị cấm")
        status = "LOẠI"
        reason = " | ".join(reasons) if reasons else "Không đạt tiêu chí"

    print(f"\nKẾT QUẢ: {status}")
    print(f"Ngành dự đoán: {category}  ({confidence:.1f}%)")
    print(f"Lý do: {reason}")
    print("\nXác suất tất cả ngành:")
    for cat, p in sorted(all_probs.items(), key=lambda x: -x[1]):
        print(f"  {cat:<30} {p:.1f}%")

    return {
        "status":       status,
        "reason":       reason,
        "category":     category,
        "confidence":   confidence,
        "all_probs":    all_probs,
        "logic_result": logic_result,
        "cleaned_text": cleaned,
    }


if __name__ == "__main__":
    CV_PATH = "Data/Test_CVs/6_nganh_chinh/CV_INFORMATION-TECHNOLOGY.pdf"
    REQUIRED = ["python", "sql"]

    if os.path.exists(CV_PATH):
        run_prediction(CV_PATH, required_skills=REQUIRED)
    else:
        print(f"Không tìm thấy: {CV_PATH}")
