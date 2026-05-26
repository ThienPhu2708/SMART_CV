"""
Tích hợp dataset Neuralframe AI (Bitfest 2025) vào dataset gốc.
Đặt file tải về tại: Data/Raw/neuralframe_resumes.csv

Quy trình:
  1. Đọc dataset mới (35 cột có cấu trúc)
  2. Dựng resume text từ các cột: career_objective + skills + degree + company
  3. Gán Category bằng rule-based keyword mapping
  4. Chỉ giữ các bản ghi có độ tin cậy cao (score >= 2)
  5. Gộp với Resume.csv gốc → Resume_merged.csv
"""

import ast
import os
import re
import pandas as pd

# ── Mapping kỹ năng / từ khóa → Category ─────────────────────────────────────
# Mỗi category có list các từ khóa; bản ghi được gán label nếu khớp nhiều nhất
KEYWORD_MAP = {
    "INFORMATION-TECHNOLOGY": [
        "python", "java", "javascript", "c++", "sql", "machine learning",
        "deep learning", "data science", "tensorflow", "pytorch", "react",
        "angular", "node", "software development", "software engineer",
        "backend", "frontend", "devops", "cloud", "aws", "azure", "linux",
        "docker", "kubernetes", "network", "cybersecurity", "database",
        "programming", "algorithm", "it support", "web developer",
        "mobile developer", "android", "ios", "flutter",
    ],
    "ACCOUNTANT": [
        "accounting", "accountant", "bookkeeping", "ledger", "tally",
        "cpa", "cma", "audit", "taxation", "gst", "accounts payable",
        "accounts receivable", "financial statement", "balance sheet",
        "reconciliation",
    ],
    "FINANCE": [
        "finance", "financial analysis", "investment", "portfolio",
        "equity", "risk management", "treasury", "bloomberg", "cfa",
        "corporate finance", "financial modeling", "valuation",
        "banking", "credit analysis",
    ],
    "ENGINEERING": [
        "mechanical engineering", "electrical engineering",
        "civil engineering", "structural", "autocad", "solidworks",
        "manufacturing", "production", "quality control", "six sigma",
        "plc", "automation", "embedded", "mechatronics",
    ],
    "CONSTRUCTION": [
        "construction", "site engineer", "project management", "quantity survey",
        "rcc", "concrete", "blueprint", "estimation", "tendering",
        "civil", "surveying", "infrastructure",
    ],
    "HEALTHCARE": [
        "doctor", "physician", "nurse", "nursing", "medical", "clinical",
        "hospital", "patient", "surgery", "pharmacy", "pharmacist",
        "laboratory", "radiology", "mbbs", "md", "healthcare",
        "physiotherapy", "occupational therapy",
    ],
    "TEACHER": [
        "teaching", "teacher", "professor", "lecturer", "educator",
        "curriculum", "pedagogy", "school", "university", "tutor",
        "instruction", "lesson plan", "academic",
    ],
    "BUSINESS-DEVELOPMENT": [
        "business development", "business analyst", "strategy",
        "stakeholder", "market research", "partnership", "growth",
        "expansion", "b2b", "crm", "kpi", "consultant",
        "management consulting",
    ],
    "SALES": [
        "sales", "salesperson", "revenue", "target", "quota",
        "lead generation", "cold calling", "negotiation", "closing",
        "retail", "fmcg", "direct sales", "sales executive",
    ],
    "HR": [
        "human resources", "hr", "recruitment", "talent acquisition",
        "onboarding", "payroll", "employee relations", "hris",
        "performance management", "workforce planning",
    ],
    "BANKING": [
        "banking", "bank", "loan", "credit", "mortgage",
        "financial services", "retail banking", "teller",
        "branch manager", "clearing", "compliance",
    ],
    "CONSULTANT": [
        "consulting", "management consultant", "advisory",
        "solution architect", "erp", "sap", "oracle",
    ],
    "DESIGNER": [
        "graphic design", "ui design", "ux design", "adobe",
        "photoshop", "illustrator", "figma", "sketch",
        "branding", "typography", "visual design", "animation",
        "3d modeling", "blender",
    ],
    "DIGITAL-MEDIA": [
        "digital marketing", "seo", "sem", "social media",
        "content marketing", "google analytics", "email marketing",
        "ppc", "influencer", "content creation", "youtube",
        "video editing", "copywriting",
    ],
    "PUBLIC-RELATIONS": [
        "public relations", "pr", "media relations", "press release",
        "communications", "brand management", "event management",
        "crisis management", "journalism",
    ],
    "AVIATION": [
        "aviation", "pilot", "airline", "aircraft", "atc",
        "flight", "cabin crew", "aerospace", "aeronautical",
    ],
    "AGRICULTURE": [
        "agriculture", "farming", "horticulture", "agronomy",
        "crop", "soil", "irrigation", "fertilizer", "livestock",
        "veterinary", "food science", "food technology",
    ],
    "FITNESS": [
        "fitness", "personal trainer", "gym", "nutrition",
        "sports", "athlete", "yoga", "wellness", "physical training",
    ],
    "ARTS": [
        "fine arts", "painting", "sculpture", "music",
        "photography", "art", "gallery", "creative arts", "craft",
    ],
    "CHEF": [
        "chef", "cook", "culinary", "kitchen", "food preparation",
        "pastry", "restaurant", "cuisine", "catering", "barista",
    ],
    "ADVOCATE": [
        "lawyer", "advocate", "attorney", "legal", "law",
        "litigation", "contract", "compliance", "judiciary",
        "paralegal", "barrister",
    ],
    "APPAREL": [
        "fashion", "apparel", "clothing", "textile", "merchandising",
        "garment", "retail fashion", "buyer", "fashion design",
    ],
    "AUTOMOBILE": [
        "automobile", "automotive", "car", "vehicle", "mechanic",
        "auto repair", "engine", "transmission", "service technician",
    ],
    "BPO": [
        "bpo", "business process outsourcing", "call center",
        "customer service", "customer support", "help desk",
        "data entry", "back office", "voice process",
        "non voice", "kpo",
    ],
}


def _safe_parse_list(val):
    """Parse string dạng "['a', 'b']" hoặc trả về list rỗng."""
    if not val or (isinstance(val, float)):
        return []
    if isinstance(val, list):
        return val
    try:
        parsed = ast.literal_eval(str(val))
        return parsed if isinstance(parsed, list) else [str(parsed)]
    except Exception:
        return [str(val)]


def _build_resume_text(row):
    """Tổng hợp text từ các cột có cấu trúc."""
    parts = []

    obj = str(row.get("career_objective", "") or "")
    if obj and obj.lower() not in ("nan", "none", "n/a", ""):
        parts.append(obj)

    skills = _safe_parse_list(row.get("skills"))
    if skills:
        parts.append("Skills: " + " ".join(str(s) for s in skills))

    degrees = _safe_parse_list(row.get("degree_names"))
    majors  = _safe_parse_list(row.get("major_field_of_studies"))
    if degrees:
        parts.append("Education: " + " ".join(str(d) for d in degrees + majors))

    companies = _safe_parse_list(row.get("professional_company_names"))
    real_companies = [
        c for c in companies
        if c and str(c).lower() not in ("company name", "nan", "none", "n/a")
    ]
    if real_companies:
        parts.append("Experience: " + " ".join(str(c) for c in real_companies))

    return " ".join(parts).strip()


def _assign_category(text):
    """
    Gán Category dựa trên số từ khóa khớp.
    Trả về (category, score) — score = số từ khóa tìm thấy của category thắng.
    """
    text_lower = text.lower()
    scores = {}
    for cat, keywords in KEYWORD_MAP.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[cat] = score

    if not scores:
        return None, 0

    best_cat   = max(scores, key=scores.get)
    best_score = scores[best_cat]

    # Kiểm tra ambiguity: nếu có tie hoặc second-best quá gần
    sorted_scores = sorted(scores.values(), reverse=True)
    second = sorted_scores[1] if len(sorted_scores) > 1 else 0
    if best_score > 0 and best_score == second:
        return None, 0  # ambiguous → bỏ qua

    return best_cat, best_score


def merge_datasets(
    new_csv_path  = "Data/Raw/neuralframe_resumes.csv",
    base_csv_path = "Data/Raw/Resume.csv",
    output_path   = "Data/Raw/Resume_merged.csv",
    min_score     = 2,
):
    """
    Gộp dataset mới vào dataset gốc.

    Args:
        min_score: Số từ khóa khớp tối thiểu để giữ bản ghi (lọc noise).
    """
    if not os.path.exists(new_csv_path):
        print(f"Loi: Khong tim thay {new_csv_path}")
        print("Hay tai dataset ve va dat o Data/Raw/neuralframe_resumes.csv")
        return

    print(f"Doc dataset moi: {new_csv_path}")
    df_new = pd.read_csv(new_csv_path, encoding="utf-8", low_memory=False)
    print(f"  {len(df_new)} ban ghi | {df_new.shape[1]} cot")

    # Xây dựng text và gán nhãn
    print("Dang tong hop resume text va gan nhan Category...")
    records = []
    skipped_no_text   = 0
    skipped_low_score = 0
    skipped_ambiguous = 0

    for idx, row in df_new.iterrows():
        text = _build_resume_text(row)
        if len(text.split()) < 10:
            skipped_no_text += 1
            continue

        category, score = _assign_category(text)
        if category is None:
            skipped_ambiguous += 1
            continue
        if score < min_score:
            skipped_low_score += 1
            continue

        records.append({
            "ID":           f"NF_{idx}",
            "Resume_str":   text,
            "Category":     category,
        })

    df_converted = pd.DataFrame(records)
    print(f"\nKet qua chuyen doi:")
    print(f"  Giu lai   : {len(df_converted)} ban ghi")
    print(f"  Bo qua (it text)  : {skipped_no_text}")
    print(f"  Bo qua (score<{min_score}): {skipped_low_score}")
    print(f"  Bo qua (ambiguous): {skipped_ambiguous}")

    if df_converted.empty:
        print("Khong co ban ghi nao du dieu kien. Kiem tra ten cot trong file CSV moi.")
        return

    # Phân bố category mới
    print("\nPhan bo Category trong dataset moi (sau loc):")
    dist = df_converted["Category"].value_counts()
    for cat, cnt in dist.items():
        print(f"  {cat:<25} {cnt}")

    # Gộp với dataset gốc
    df_base = pd.read_csv(base_csv_path)[["ID", "Resume_str", "Category"]]
    df_base["ID"] = df_base["ID"].astype(str)

    df_merged = pd.concat([df_base, df_converted], ignore_index=True)
    df_merged.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\nSau khi gop:")
    print(f"  Dataset goc : {len(df_base)} ban ghi")
    print(f"  Dataset moi : {len(df_converted)} ban ghi")
    print(f"  Tong cong   : {len(df_merged)} ban ghi")
    print(f"\nPhan bo Category sau khi gop (chi liet ke thay doi):")
    for cat in sorted(df_merged["Category"].unique()):
        n_base   = (df_base["Category"] == cat).sum()
        n_merged = (df_merged["Category"] == cat).sum()
        delta    = n_merged - n_base
        flag     = " <-- tang them" if delta > 0 else ""
        print(f"  {cat:<25} {n_base:>4} -> {n_merged:>4} (+{delta}){flag}")

    print(f"\nDa luu: {output_path}")
    print("Tiep theo: cap nhat duong dan trong main.py thanh Data/Raw/Resume_merged.csv")


if __name__ == "__main__":
    merge_datasets()
