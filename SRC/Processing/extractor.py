import easyocr
import PyPDF2

reader_ocr = easyocr.Reader(['en'])

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

def extract_text_from_image(image_path):
    result = reader_ocr.readtext(
        image_path,
        detail=0
    )
    text = " ".join(result)
    return text