import easyocr
import PyPDF2
from pdf2image import convert_from_path
reader_ocr = easyocr.Reader(['en'])
# =========================
# PDF text extraction
# =========================
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
    except Exception as e:
        print("PDF reading error:", e)
    return text
# =========================
# OCR image
# =========================
def extract_text_from_image(image_path):
    result = reader_ocr.readtext(
        image_path,
        detail=0
    )
    text = " ".join(result)
    return text
# =========================
# OCR scanned PDF
# =========================
def extract_text_from_scanned_pdf(pdf_path):
    text = ""
    try:
        pages = convert_from_path(pdf_path)
        for i, page in enumerate(pages):
            image_name = f"temp_page_{i}.jpg"
            page.save(image_name, "JPEG")
            result = reader_ocr.readtext(
                image_name,
                detail=0
            )
            page_text = " ".join(result)
            text += page_text + "\n"
    except Exception as e:
        print("OCR PDF error:", e)
    return text
# =========================
# Main pipeline
# =========================
def extract_text(file_path):
    # thử đọc PDF thường trước
    text = extract_text_from_pdf(file_path)
    # nếu text quá ít => nghi là PDF scan
    if len(text.strip()) < 20:
        print("Scanned PDF detected. Using OCR...")
        text = extract_text_from_scanned_pdf(file_path)
    return text