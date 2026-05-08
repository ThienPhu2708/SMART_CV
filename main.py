from SRC.Processing.extractor import extract_text_from_pdf
from SRC.Processing.cleaner import clean_text
text = extract_text_from_pdf("Data/Raw/cv.pdf")

cleaned_text = clean_text(text)

print(cleaned_text)