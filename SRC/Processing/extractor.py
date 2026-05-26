import os
import PyPDF2

_ocr_reader = None


def _get_ocr_reader():
    """Lazy-init EasyOCR với tiếng Anh + tiếng Việt."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        try:
            # en + vi để nhận cả keyword tiếng Anh lẫn text tiếng Việt
            _ocr_reader = easyocr.Reader(['en', 'vi'], verbose=False)
        except Exception:
            # fallback chỉ en nếu model vi chưa download được
            _ocr_reader = easyocr.Reader(['en'], verbose=False)
    return _ocr_reader


def extract_text_with_pdfplumber(file_path):
    """pdfplumber — xử lý tốt PDF có layout phức tạp, sidebar, multi-column."""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                words = page.extract_words(
                    x_tolerance=3,
                    y_tolerance=3,
                    keep_blank_chars=False,
                    use_text_flow=False,
                )
                if words:
                    # Sắp xếp theo dòng (top), rồi theo cột (x0) — bắt cả 2 cột
                    words_sorted = sorted(
                        words, key=lambda w: (round(w["top"] / 5) * 5, w["x0"])
                    )
                    lines, current_line, last_top = [], [], None
                    for w in words_sorted:
                        top = round(w["top"] / 5) * 5
                        if last_top is not None and abs(top - last_top) > 5:
                            lines.append(" ".join(current_line))
                            current_line = []
                        current_line.append(w["text"])
                        last_top = top
                    if current_line:
                        lines.append(" ".join(current_line))
                    text += "\n".join(lines) + "\n"
                else:
                    text += (page.extract_text() or "") + "\n"
        return text.strip()
    except ImportError:
        return ""
    except Exception as e:
        print(f"pdfplumber error: {e}")
        return ""


def extract_text_with_pypdf2(file_path):
    """PyPDF2 — fallback cho PDF text 1 cột đơn giản."""
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
    except Exception as e:
        print(f"PyPDF2 error: {e}")
    return text.strip()


def _preprocess_image(img):
    """
    Tiền xử lý ảnh trước khi OCR:
    - Chuyển grayscale → giảm nhiễu màu nền
    - Tăng contrast → chữ rõ hơn so với background
    - Sharpen → cạnh chữ sắc hơn
    """
    from PIL import ImageEnhance, ImageFilter
    img = img.convert("L")                          # grayscale
    img = ImageEnhance.Contrast(img).enhance(2.0)   # tăng contrast x2
    img = img.filter(ImageFilter.SHARPEN)            # sharpen
    return img


def _pdf_to_images_fitz(file_path, dpi=300):
    """
    Dùng PyMuPDF (fitz) render PDF thành list PIL Image.
    Không cần Poppler — chạy thẳng trên Windows.
    DPI=300 đủ rõ để OCR nhận font nhỏ trong CV thiết kế.
    """
    import fitz
    from PIL import Image
    import io

    doc = fitz.open(file_path)
    images = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    doc.close()
    return images


def extract_text_from_scanned_pdf(file_path):
    """
    OCR từng trang PDF scan.
    Dùng PyMuPDF render ảnh (300 DPI) + tiền xử lý contrast → EasyOCR.
    """
    text = ""
    temp_files = []
    try:
        images = _pdf_to_images_fitz(file_path, dpi=300)
        reader = _get_ocr_reader()
        for i, img in enumerate(images):
            img = _preprocess_image(img)
            tmp = f"temp_page_{i}.jpg"
            img.save(tmp, "JPEG", quality=95)
            temp_files.append(tmp)
            result = reader.readtext(tmp, detail=0)
            text += " ".join(result) + "\n"
    except Exception as e:
        print(f"OCR error: {e}")
        raise   # re-raise để UI debug thấy được lỗi
    finally:
        for f in temp_files:
            try:
                os.remove(f)
            except OSError:
                pass
    return text.strip()


def extract_text(file_path):
    """
    Pipeline trích xuất văn bản — thử theo thứ tự:
    1. pdfplumber  — PDF có layout phức tạp, multi-column
    2. PyPDF2      — PDF text đơn giản 1 cột
    3. OCR         — PDF scan (ảnh thuần túy, không có text layer)
    """
    text = extract_text_with_pdfplumber(file_path)

    if len(text.strip()) < 50:
        text = extract_text_with_pypdf2(file_path)

    if len(text.strip()) < 50:
        print("Scanned PDF detected. Using OCR...")
        text = extract_text_from_scanned_pdf(file_path)

    return text
