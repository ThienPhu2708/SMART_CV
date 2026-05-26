import os
import pandas as pd

from SRC.Processing.cleaner import clean_text
from SRC.Processing.vectorizer import create_tfidf_vectors
from SRC.logic_gates import check_keyword, logic_and

# ── Cấu hình ──────────────────────────────────────────────────────────────────
MERGED_CSV  = "Data/Raw/Resume_merged.csv"
ORIGINAL_CSV = "Data/Raw/Resume.csv"
MAX_NEW_PER_CLASS = 200   # tối đa mẫu mới (NF_*) thêm vào mỗi class

# =========================
# LOAD & CÂN BẰNG DATASET
# =========================
SYNTHETIC_CSV = "Data/Raw/synthetic_resumes.csv"

if os.path.exists(MERGED_CSV):
    df_all = pd.read_csv(MERGED_CSV)[["ID", "Resume_str", "Category"]]
    df_all["ID"] = df_all["ID"].astype(str)

    # Tách mẫu gốc và mẫu từ Neuralframe (NF_)
    df_orig = df_all[~df_all["ID"].str.startswith("NF_")].copy()
    df_nf   = df_all[df_all["ID"].str.startswith("NF_")].copy()

    # Giới hạn Neuralframe tối đa MAX_NEW_PER_CLASS/class
    nf_parts = []
    for cat, grp in df_nf.groupby("Category"):
        nf_parts.append(grp.sample(n=min(len(grp), MAX_NEW_PER_CLASS), random_state=42))
    df_nf_bal = pd.concat(nf_parts, ignore_index=True) if nf_parts else pd.DataFrame()

    df = pd.concat([df_orig, df_nf_bal], ignore_index=True)
else:
    df   = pd.read_csv(ORIGINAL_CSV)[["ID", "Resume_str", "Category"]]
    df_orig = df.copy()
    df_nf_bal = pd.DataFrame()

# Gộp thêm synthetic data (SYN_*)
if os.path.exists(SYNTHETIC_CSV):
    df_syn = pd.read_csv(SYNTHETIC_CSV)[["ID", "Resume_str", "Category"]]
    df     = pd.concat([df, df_syn], ignore_index=True)
    n_syn  = len(df_syn)
else:
    n_syn = 0

n_nf  = len(df_nf_bal) if "df_nf_bal" in dir() else 0
n_orig = len(df_orig)
print(f"Dataset loaded: {n_orig} goc + {n_nf} Neuralframe + {n_syn} synthetic = {len(df)} tong")
print("Phan bo sau khi gop:")
dist = df["Category"].value_counts()
for cat, cnt in dist.items():
    print(f"  {cat:<25} {cnt}")

# =========================
# CLEAN TEXT
# =========================
print("\nDang lam sach van ban...")
df["cleaned_resume"] = df["Resume_str"].apply(clean_text)
df = df[df["cleaned_resume"].str.strip() != ""]
print(f"Text cleaning completed! Con lai: {len(df)} ban ghi")

# =========================
# SAVE CLEAN DATA
# =========================
os.makedirs("Data/Processed", exist_ok=True)
cleaned_df = df[["ID", "Category", "cleaned_resume"]]
cleaned_df.to_csv("Data/Processed/cleaned_resume.csv", index=False)
print("Cleaned dataset saved!")

# =========================
# TF-IDF
# =========================
create_tfidf_vectors()

# =========================
# LOGIC GATES TEST
# =========================
sample_text = "python django sql"
python_skill = check_keyword(sample_text, "python")
django_skill = check_keyword(sample_text, "django")
result = logic_and(python_skill, django_skill)
print("AND Result:", result)
