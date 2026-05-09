import pandas as pd

from SRC.Processing.cleaner import clean_text
from SRC.Processing.vectorizer import create_tfidf_vectors
from SRC.Processing.logic_helper import *

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("Data/Raw/Resume.csv")
# giữ cột cần thiết
df = df[[
    "ID",
    "Resume_str",
    "Category"
]]
print("Dataset loaded!")
# =========================
# CLEAN TEXT
# =========================
df["cleaned_resume"] = df[
    "Resume_str"
].apply(clean_text)
# xóa text rỗng
df = df[
    df["cleaned_resume"].str.strip() != ""
]
print("Text cleaning completed!")
# =========================
# SAVE CLEAN DATA
# =========================
cleaned_df = df[
    ["ID", "Category", "cleaned_resume"]
]
cleaned_df.to_csv(
    "Data/Processed/cleaned_resume.csv",
    index=False
)
print("Cleaned dataset saved!")
# =========================
# TF-IDF
# =========================
create_tfidf_vectors()
# =========================
# LOGIC GATES TEST
# =========================
sample_text = "python django sql"
python_skill = check_keyword(
    sample_text,
    "python"
)
django_skill = check_keyword(
    sample_text,
    "django"
)
result = logic_and(
    python_skill,
    django_skill
)
print("AND Result:", result)