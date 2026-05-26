_lg_lemmatizer = None


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


def check_mandatory_skills(resume_text, required_skills):
    """
    ỨNG DỤNG LOGIC AND: CV phải có ĐỦ tất cả kỹ năng bắt buộc.
    Ví dụ: phải có 'Python' AND 'SQL'.
    """
    if not required_skills:
        return 1
    status = 1
    for skill in required_skills:
        status = logic_and(status, check_keyword(resume_text, skill))
    return status


def check_optional_skills(resume_text, optional_skills):
    """
    ỨNG DỤNG LOGIC OR: CV có ÍT NHẤT một trong các kỹ năng ưu tiên.
    Ví dụ: có 'React' OR 'Vue' OR 'Angular'.
    """
    if not optional_skills:
        return 0
    status = 0
    for skill in optional_skills:
        status = logic_or(status, check_keyword(resume_text, skill))
    return status


def check_blacklist_keywords(resume_text, blacklist):
    """
    ỨNG DỤNG LOGIC NOT: CV bị loại nếu chứa bất kỳ từ khóa cấm nào.
    Trả về 1 (pass) nếu KHÔNG có từ khóa nào trong blacklist.
    Ví dụ: loại nếu CV đề cập 'fraud', 'terminated'.
    """
    if not blacklist:
        return 1
    for word in blacklist:
        if check_keyword(resume_text, word):
            return 0    # NOT: phát hiện từ cấm → loại
    return 1


def check_exclusive_skills(resume_text, skill_a, skill_b):
    """
    ỨNG DỤNG LOGIC XOR: CV có đúng MỘT trong hai kỹ năng (không thiếu, không thừa).
    Ví dụ: cần 'PyTorch' XOR 'TensorFlow' (chuyên sâu một framework).
    """
    has_a = check_keyword(resume_text, skill_a)
    has_b = check_keyword(resume_text, skill_b)
    return logic_xor(has_a, has_b)


def evaluate_candidate(resume_text, required_skills=None, optional_skills=None,
                        blacklist=None, exclusive_pair=None):
    """
    Đánh giá tổng hợp ứng viên qua tất cả các cổng logic.
    Trả về dict với kết quả từng cổng và trạng thái cuối.
    """
    required_skills = required_skills or []
    optional_skills = optional_skills or []
    blacklist = blacklist or []

    mandatory_pass = check_mandatory_skills(resume_text, required_skills)
    optional_pass = check_optional_skills(resume_text, optional_skills)
    blacklist_pass = check_blacklist_keywords(resume_text, blacklist)
    exclusive_pass = (
        check_exclusive_skills(resume_text, exclusive_pair[0], exclusive_pair[1])
        if exclusive_pair and len(exclusive_pair) == 2
        else None
    )

    # Điều kiện bắt buộc để qua: mandatory AND NOT blacklist
    passed = bool(mandatory_pass and blacklist_pass)

    return {
        "mandatory_pass": mandatory_pass,
        "optional_pass": optional_pass,
        "blacklist_pass": blacklist_pass,
        "exclusive_pass": exclusive_pass,
        "overall_pass": passed,
    }
