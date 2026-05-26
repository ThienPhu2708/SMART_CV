"""
Tạo PDF test từ các mẫu trong Resume.csv gốc.
Dùng PyMuPDF (fitz) — đã có sẵn trong requirements.
"""
import os, random
import pandas as pd
import fitz   # PyMuPDF

random.seed(7)

CATEGORIES = [
    "INFORMATION-TECHNOLOGY", "HEALTHCARE", "FINANCE",
    "ENGINEER", "TEACHER", "CHEF", "AGRICULTURE",
    "BPO", "AUTOMOBILE", "ARTS", "FITNESS", "DESIGNER",
]

OUTPUT_DIR = "Data/Test_CVs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("Data/Raw/Resume.csv")[["Category", "Resume_str"]]


def create_pdf(text: str, output_path: str):
    doc  = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4

    # Margins
    x0, y0 = 50, 60
    max_w   = 495   # width available
    font    = "helv"
    size    = 10
    line_h  = 14

    # Header bar
    page.draw_rect(fitz.Rect(0, 0, 595, 45), color=(0.18, 0.36, 0.60), fill=(0.18, 0.36, 0.60))
    page.insert_text((x0, 28), "CURRICULUM VITAE", fontname=font, fontsize=14, color=(1, 1, 1))

    y = y0
    for raw_line in text.split("\n"):
        # Wrap dài
        words = raw_line.split()
        line  = ""
        for word in words:
            test = (line + " " + word).strip()
            # Ước lượng chiều rộng: mỗi ký tự ~5.5px với size 10
            if len(test) * 5.5 > max_w:
                if y > 800:
                    page = doc.new_page(width=595, height=842)
                    y    = 60
                page.insert_text((x0, y), line, fontname=font, fontsize=size, color=(0, 0, 0))
                y    += line_h
                line  = word
            else:
                line = test
        if line:
            if y > 800:
                page = doc.new_page(width=595, height=842)
                y    = 60
            page.insert_text((x0, y), line, fontname=font, fontsize=size, color=(0, 0, 0))
            y += line_h
        else:
            y += 4   # blank line

    doc.save(output_path)
    doc.close()


created = []
for cat in df["Category"].unique():
    group   = df[df["Category"] == cat]
    sample  = group.sample(1, random_state=random.randint(0, 9999)).iloc[0]
    text    = str(sample["Resume_str"])
    safe    = cat.replace("/", "-").replace(" ", "_")
    out     = os.path.join(OUTPUT_DIR, f"CV_{safe}.pdf")
    create_pdf(text, out)
    created.append((cat, out))
    print(f"  [{cat:<25}] -> {out}")

print(f"\nDa tao {len(created)} file PDF tai: {OUTPUT_DIR}/")
print("Su dung cac file nay de test tren UI.")
