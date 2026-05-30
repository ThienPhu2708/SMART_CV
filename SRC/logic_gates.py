"""
Logic Gates cho hệ thống SmartCV.

Kiểm tra kỹ năng trong CV dùng Hybrid approach:
  1. check_keyword()    — exact/lemma/fuzzy match (nhanh, độ chính xác cao)
  2. check_skill_bert() — BERT cosine similarity (fallback khi keyword không tìm thấy)

BERT là mô hình đã HỌC ngữ nghĩa → skill detection thông minh hơn exact match.
Ví dụ: "scripting with data wrangling" → BERT hiểu liên quan Python dù không có chữ "python".
"""
import numpy as np

_lg_lemmatizer = None
_bert_skill_cache: dict = {}   # cache embedding của skill keywords để tránh encode lại


def _get_lg_lemmatizer():
    global _lg_lemmatizer
    if _lg_lemmatizer is None:
        import nltk
        try:
            from nltk.stem import WordNetLemmatizer
            lem = WordNetLemmatizer()
            lem.lemmatize("test")
            _lg_lemmatizer = lem
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            from nltk.stem import WordNetLemmatizer
            _lg_lemmatizer = WordNetLemmatizer()
    return _lg_lemmatizer


def logic_and(a, b):
    """AND gate: trả về 1 nếu cả hai điều kiện đều thỏa."""
    return 1 if a and b else 0


def logic_or(a, b):
    """OR gate: trả về 1 nếu ít nhất một điều kiện thỏa."""
    return 1 if a or b else 0


def logic_not(a):
    """NOT gate: đảo ngược giá trị boolean."""
    return 0 if a else 1


def logic_xor(a, b):
    """XOR gate: trả về 1 nếu đúng một trong hai điều kiện thỏa (không cùng lúc)."""
    return 1 if bool(a) != bool(b) else 0


def check_keyword(text, keyword):
    """
    Kiểm tra keyword trong text.
    - Exact match trước (nhanh, không có false positive)
    - Thử lemmatized keyword vì text đã qua clean_text (WordNet lemmatizer).
      Ví dụ: "Technical Sales" -> "technical sale" để khớp với text đã lemmatized.
    - Fuzzy match để xử lý OCR đọc sai vài ký tự (chỉ cho keyword đơn từ).
      Ngưỡng fuzzy phụ thuộc độ dài keyword:
        ≤3 ký tự (sql, go) : bỏ qua fuzzy — dễ false positive
        4-5 ký tự (java, ruby): threshold 0.75
        ≥6 ký tự (python, django): threshold 0.82
    """
    if not text or not isinstance(text, str):
        return 0
    kw = keyword.lower()
    txt = text.lower()

    if kw in txt:
        return 1

    # text đã lemmatized bởi clean_text — thử lemmatize keyword trước khi tìm
    lemmatizer = _get_lg_lemmatizer()
    kw_lem = " ".join(lemmatizer.lemmatize(w) for w in kw.split())
    if kw_lem != kw and kw_lem in txt:
        return 1

    n = len(kw)
    if n <= 3:
        return 0   # keyword quá ngắn, không fuzzy

    threshold = 0.75 if n <= 5 else 0.82

    from difflib import SequenceMatcher
    for word in txt.split():
        if abs(len(word) - n) <= 2:   # chỉ so sánh word có độ dài tương đương
            ratio = SequenceMatcher(None, kw, word).ratio()
            if ratio >= threshold:
                return 1

    return 0


def check_skill_bert(cv_embedding: np.ndarray, skill_keyword: str,
                     threshold: float = 0.28) -> tuple[float, int]:
    """
    Kiểm tra kỹ năng bằng BERT cosine similarity.

    BERT đã HỌC ngữ nghĩa → phát hiện kỹ năng thông minh hơn exact match:
    - "scripting language for automation" → match "Python" dù không có chữ python
    - "proficient in JS frameworks"       → match "JavaScript"
    - OCR lỗi "Pythan"                    → BERT embedding vẫn gần "Python"

    Args:
        cv_embedding : vector 384-dim từ BERT encode toàn bộ CV (đã L2-normalized)
        skill_keyword: tên kỹ năng HR nhập, ví dụ "Python", "Machine Learning"
        threshold    : ngưỡng cosine similarity (mặc định 0.28 — đủ chặt tránh false positive)

    Returns:
        (score, match): score là float cosine sim, match là 0 hoặc 1
    """
    global _bert_skill_cache
    if cv_embedding is None:
        return 0.0, 0

    # Encode skill keyword dưới dạng câu ngữ cảnh — giúp BERT hiểu đúng nghĩa
    cache_key = skill_keyword.lower().strip()
    if cache_key not in _bert_skill_cache:
        try:
            from SRC.Processing.bert_encoder import encode_single
            # Mô tả kỹ năng dưới dạng câu để BERT encode đúng ngữ cảnh
            skill_phrases = [
                f"experience with {skill_keyword}",
                f"{skill_keyword} skills and knowledge",
                f"proficient in {skill_keyword}",
            ]
            # Trung bình embedding của 3 cách diễn đạt → embedding ổn định hơn
            embs = [encode_single(p).flatten() for p in skill_phrases]
            skill_emb = np.mean(embs, axis=0)
            norm = np.linalg.norm(skill_emb)
            skill_emb = skill_emb / (norm + 1e-8)
            _bert_skill_cache[cache_key] = skill_emb
        except Exception:
            return 0.0, 0

    skill_emb = _bert_skill_cache[cache_key]
    cv_emb    = cv_embedding.flatten()

    # Cosine similarity (cả hai đã L2-normalized nên chỉ cần dot product)
    score = float(np.dot(cv_emb, skill_emb) /
                  (np.linalg.norm(cv_emb) * np.linalg.norm(skill_emb) + 1e-8))
    return score, int(score >= threshold)


def check_keyword_hybrid(text: str, keyword: str,
                         cv_embedding: np.ndarray = None,
                         bert_threshold: float = 0.28) -> int:
    """
    Hybrid skill check: exact/fuzzy match trước, BERT similarity làm fallback.

    Luồng xử lý:
      1. Exact / lemma / fuzzy match (nhanh, không false positive)
         → tìm thấy: trả về 1 ngay
      2. BERT cosine similarity (chỉ khi step 1 không tìm thấy)
         → score >= bert_threshold: trả về 1
         → score < threshold: trả về 0

    HR workflow không thay đổi — thêm BERT chỉ là engine bên trong.
    """
    # Bước 1: rule-based (giữ nguyên logic cũ)
    if check_keyword(text, keyword):
        return 1

    # Bước 2: BERT semantic fallback
    if cv_embedding is not None:
        _, bert_match = check_skill_bert(cv_embedding, keyword, bert_threshold)
        return bert_match

    return 0


def check_mandatory_skills(resume_text, required_skills, cv_embedding=None):
    """
    ỨNG DỤNG LOGIC AND: CV phải có ĐỦ tất cả kỹ năng bắt buộc.
    Ví dụ: phải có 'Python' AND 'SQL'.
    Dùng hybrid check: exact/fuzzy trước, BERT fallback nếu không tìm thấy.
    """
    if not required_skills:
        return 1
    status = 1
    for skill in required_skills:
        found = check_keyword_hybrid(resume_text, skill, cv_embedding)
        status = logic_and(status, found)
    return status


def check_optional_skills(resume_text, optional_skills, cv_embedding=None):
    """
    ỨNG DỤNG LOGIC OR: CV có ÍT NHẤT một trong các kỹ năng ưu tiên.
    Ví dụ: có 'React' OR 'Vue' OR 'Angular'.
    Dùng hybrid check: exact/fuzzy trước, BERT fallback nếu không tìm thấy.
    """
    if not optional_skills:
        return 0
    status = 0
    for skill in optional_skills:
        found = check_keyword_hybrid(resume_text, skill, cv_embedding)
        status = logic_or(status, found)
    return status


def check_blacklist_keywords(resume_text, blacklist, cv_embedding=None):
    """
    ỨNG DỤNG LOGIC NOT: CV bị loại nếu chứa bất kỳ từ khóa cấm nào.
    Trả về 1 (pass) nếu KHÔNG có từ khóa nào trong blacklist.
    Ví dụ: loại nếu CV đề cập 'fraud', 'terminated'.
    NOT gate dùng threshold cao hơn (0.35) để tránh loại nhầm.
    """
    if not blacklist:
        return 1
    for word in blacklist:
        # Blacklist dùng threshold cao hơn để tránh false positive
        if check_keyword(resume_text, word):
            return 0
        if cv_embedding is not None:
            _, bert_match = check_skill_bert(cv_embedding, word, threshold=0.40)
            if bert_match:
                return 0
    return 1


def check_exclusive_skills(resume_text, skill_a, skill_b, cv_embedding=None):
    """
    ỨNG DỤNG LOGIC XOR: CV có đúng MỘT trong hai kỹ năng (không thiếu, không thừa).
    Ví dụ: cần 'PyTorch' XOR 'TensorFlow' (chuyên sâu một framework).
    Dùng hybrid check: exact/fuzzy trước, BERT fallback nếu không tìm thấy.
    """
    has_a = check_keyword_hybrid(resume_text, skill_a, cv_embedding)
    has_b = check_keyword_hybrid(resume_text, skill_b, cv_embedding)
    return logic_xor(has_a, has_b)


def evaluate_candidate(resume_text, required_skills=None, optional_skills=None,
                        blacklist=None, exclusive_pair=None,
                        cv_embedding=None):
    """
    Đánh giá tổng hợp ứng viên qua tất cả các cổng logic.

    Args:
        resume_text    : văn bản CV đã qua tiền xử lý
        required_skills: danh sách kỹ năng bắt buộc (AND gate)
        optional_skills: danh sách kỹ năng ưu tiên (OR gate)
        blacklist      : danh sách từ khóa cấm (NOT gate)
        exclusive_pair : cặp kỹ năng loại trừ (XOR gate), list 2 phần tử
        cv_embedding   : vector 384-dim từ BERT — nếu có, dùng hybrid check
                         (BERT semantic fallback khi keyword không tìm thấy)

    Returns:
        dict với kết quả từng cổng và trạng thái cuối overall_pass
    """
    required_skills = required_skills or []
    optional_skills = optional_skills or []
    blacklist       = blacklist or []

    mandatory_pass = check_mandatory_skills(resume_text, required_skills, cv_embedding)
    optional_pass  = check_optional_skills(resume_text, optional_skills, cv_embedding)
    blacklist_pass = check_blacklist_keywords(resume_text, blacklist, cv_embedding)
    exclusive_pass = (
        check_exclusive_skills(resume_text,
                               exclusive_pair[0], exclusive_pair[1],
                               cv_embedding)
        if exclusive_pair and len(exclusive_pair) == 2
        else None
    )

    # Điều kiện bắt buộc: mandatory_pass AND blacklist_pass (NOT gate)
    passed = bool(mandatory_pass and blacklist_pass)

    return {
        "mandatory_pass": mandatory_pass,
        "optional_pass":  optional_pass,
        "blacklist_pass": blacklist_pass,
        "exclusive_pass": exclusive_pass,
        "overall_pass":   passed,
    }
