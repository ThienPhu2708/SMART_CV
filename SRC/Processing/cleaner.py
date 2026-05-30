import re

_stop_words  = None
_lemmatizer  = None

_VI_CHARS = re.compile(
    r'[àáạảãâầấậẩẫăặắẳẵđèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ'
    r'ÀÁẠẢÃÂẦẤẬẨẪĂẶẮẲẴĐÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸ]'
)

def _has_vietnamese(text: str) -> bool:
    """
    Trả về True chỉ khi tỉ lệ ký tự tiếng Việt vượt ngưỡng 8% tổng chữ cái.
    Tránh kích hoạt dịch thuật cho CV tiếng Anh chỉ có tên/địa chỉ tiếng Việt.
    """
    if not text:
        return False
    vi_chars = len(_VI_CHARS.findall(text))
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars == 0:
        return False
    return (vi_chars / alpha_chars) > 0.08


def translate_if_needed(text):
    """
    Phát hiện ngôn ngữ và dịch sang tiếng Anh nếu cần.
    Ưu tiên kiểm tra ký tự tiếng Việt trước langdetect để tránh nhầm lẫn
    với CV song ngữ (tiếng Việt + thuật ngữ kỹ thuật tiếng Anh).
    Fallback an toàn: nếu lỗi thư viện hoặc network thì trả nguyên text.
    """
    if not text or len(text.strip()) < 30:
        return text
    try:
        # Bước 1: kiểm tra ký tự tiếng Việt trực tiếp (độ tin cậy cao hơn)
        if _has_vietnamese(text):
            need_translate = True
        else:
            from langdetect import detect
            need_translate = detect(text[:800]) != 'en'

        if not need_translate:
            return text

        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='auto', target='en')
        # GoogleTranslator giới hạn 5000 ký tự/lần gọi
        CHUNK = 4500
        parts = [text[i:i + CHUNK] for i in range(0, len(text), CHUNK)]
        translated = " ".join(translator.translate(p) for p in parts if p.strip())
        return translated if translated.strip() else text
    except Exception:
        return text

# Từ viết tắt kỹ thuật thường gặp — chuẩn hóa trước khi xử lý
_ABBREV_MAP = {
    r"\bml\b":    "machine learning",
    r"\bai\b":    "artificial intelligence",
    r"\bnlp\b":   "natural language processing",
    r"\boop\b":   "object oriented programming",
    r"\bapi\b":   "application programming interface",
    r"\bui\b":    "user interface",
    r"\bux\b":    "user experience",
    r"\bci\b":    "continuous integration",
    r"\bcd\b":    "continuous deployment",
    r"\bdb\b":    "database",
    r"\brdbms\b": "relational database management system",
}


def _get_stop_words():
    global _stop_words
    if _stop_words is None:
        import nltk
        try:
            from nltk.corpus import stopwords
            _stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            _stop_words = set(stopwords.words('english'))
    return _stop_words


def _get_lemmatizer():
    global _lemmatizer
    if _lemmatizer is None:
        import nltk
        try:
            from nltk.stem import WordNetLemmatizer
            # Thử dùng wordnet ngay — sẽ raise LookupError nếu chưa download
            lem = WordNetLemmatizer()
            lem.lemmatize("test")
            _lemmatizer = lem
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            from nltk.stem import WordNetLemmatizer
            _lemmatizer = WordNetLemmatizer()
    return _lemmatizer


_SKILLS_HEADER = re.compile(
    r'(?:skills?|kỹ năng|competenc|expertise|technical|languages?|frameworks?|tools?)',
    re.IGNORECASE,
)
_SECTION_BREAK = re.compile(
    r'(?:experience|education|project|work history|học vấn|dự án|kinh nghiệm|'
    r'certif|achievement|award|reference|introduction|giới thiệu)',
    re.IGNORECASE,
)

def _boost_skills_section(text: str, repeat: int = 2) -> str:
    """
    Nhân đôi nội dung phần Skills để tăng trọng số cho TF-IDF.
    Giúp MLP phân biệt kỹ năng thực sự vs từ khóa xuất hiện trong mô tả dự án.
    """
    lines = text.splitlines()
    in_skills = False
    skills_lines: list[str] = []
    result_lines: list[str] = []

    for line in lines:
        if _SKILLS_HEADER.search(line):
            in_skills = True
        elif in_skills and _SECTION_BREAK.search(line):
            in_skills = False

        result_lines.append(line)
        if in_skills and line.strip():
            skills_lines.append(line)

    if skills_lines:
        # Lặp lại phần skills ở cuối để tăng tần suất token
        result_lines += skills_lines * (repeat - 1)

    return "\n".join(result_lines)


def clean_text(text):
    if not text or not isinstance(text, str):
        return ""

    # Tăng trọng số phần Skills trước khi xóa cấu trúc
    text = _boost_skills_section(text)

    text = text.lower()

    # Xóa email và URL
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)

    # Xóa số điện thoại (≥8 chữ số, có thể lẫn dấu/dấu cách)
    text = re.sub(r'\b\d[\d\s\-().+]{7,}\d\b', '', text)

    # Chuẩn hóa từ viết tắt kỹ thuật trước khi xóa ký tự đặc biệt
    for pattern, replacement in _ABBREV_MAP.items():
        text = re.sub(pattern, replacement, text)

    # Chuyển "5 years" / "3+ years" thành token duy nhất "expyears"
    # (chữ số sẽ bị xóa sau, nên dùng token cố định để TF-IDF nhận được)
    text = re.sub(r'\d+\s*\+?\s*years?', 'expyears', text)

    # Xóa các ký tự không phải chữ cái và khoảng trắng
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()

    # Lemmatization — đưa các dạng biến thể về dạng gốc
    lemmatizer = _get_lemmatizer()
    stop_words = _get_stop_words()
    words = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words and len(w) > 1
    ]

    return " ".join(words)
