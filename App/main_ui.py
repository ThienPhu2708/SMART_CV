"""
SmartCV Dashboard — Streamlit UI v2
Chạy: streamlit run App/main_ui.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import tempfile
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Safe imports ───────────────────────────────────────────────────────────────
_import_errors = []

try:
    import torch
    from SRC.model_mlp import SmartCV_MLP
    _torch_ok = True
except Exception as e:
    _torch_ok = False
    _import_errors.append(f"PyTorch / model: {e}")

try:
    from SRC.Processing.extractor import (
        extract_text_with_pdfplumber,
        extract_text_with_pypdf2,
        _pdf_to_images_fitz,
    )
    _extractor_ok = True
except Exception as e:
    _extractor_ok = False
    _import_errors.append(f"Extractor: {e}")

try:
    from SRC.Processing.cleaner import clean_text
    _cleaner_ok = True
except Exception as e:
    _cleaner_ok = False
    _import_errors.append(f"Cleaner: {e}")

try:
    from SRC.Processing.cleaner import translate_if_needed
except Exception:
    def translate_if_needed(text):
        return text

try:
    from SRC.logic_gates import evaluate_candidate, check_keyword
    _gates_ok = True
except Exception as e:
    _gates_ok = False
    _import_errors.append(f"Logic gates: {e}")

# ── Job templates ──────────────────────────────────────────────────────────────
JOB_TEMPLATES = {
    "INFORMATION-TECHNOLOGY": {
        "icon": "💻", "display": "Công nghệ thông tin",
        "required": ["python", "sql", "software development"],
        "optional": ["machine learning", "docker", "aws", "javascript", "linux"],
        "description": "Dev, Data Science, DevOps, IT...",
    },
    "ACCOUNTANT": {
        "icon": "🧾", "display": "Kế toán",
        "required": ["accounting", "audit", "financial statement"],
        "optional": ["cpa", "taxation", "bookkeeping", "excel"],
        "description": "Kế toán viên, Kiểm toán...",
    },
    "FINANCE": {
        "icon": "📈", "display": "Tài chính",
        "required": ["finance", "financial analysis", "investment"],
        "optional": ["cfa", "portfolio", "risk management", "bloomberg"],
        "description": "Phân tích tài chính, Đầu tư...",
    },
    "ENGINEERING": {
        "icon": "⚙️", "display": "Kỹ thuật",
        "required": ["engineering", "autocad", "manufacturing"],
        "optional": ["solidworks", "six sigma", "plc", "automation"],
        "description": "Cơ khí, Điện, Chế tạo...",
    },
    "CONSTRUCTION": {
        "icon": "🏗️", "display": "Xây dựng",
        "required": ["construction", "civil engineering", "project management"],
        "optional": ["autocad", "revit", "quantity surveying", "rcc"],
        "description": "Kỹ sư công trình, Giám sát...",
    },
    "HEALTHCARE": {
        "icon": "🏥", "display": "Y tế",
        "required": ["patient care", "clinical", "medical"],
        "optional": ["nursing", "pharmacy", "mbbs", "diagnosis"],
        "description": "Bác sĩ, Y tá, Dược sĩ...",
    },
    "TEACHER": {
        "icon": "🎓", "display": "Giáo dục",
        "required": ["teaching", "curriculum", "lesson plan"],
        "optional": ["pedagogy", "e-learning", "student assessment"],
        "description": "Giáo viên, Giảng viên...",
    },
    "BUSINESS-DEVELOPMENT": {
        "icon": "📊", "display": "Phát triển kinh doanh",
        "required": ["business development", "sales", "strategic planning"],
        "optional": ["crm", "b2b", "market research", "kpi"],
        "description": "BD Manager, Business Analyst...",
    },
    "SALES": {
        "icon": "🤝", "display": "Kinh doanh / Sales",
        "required": ["sales", "negotiation", "target"],
        "optional": ["cold calling", "retail", "fmcg", "salesforce"],
        "description": "Sales Executive, Account Manager...",
    },
    "HR": {
        "icon": "👥", "display": "Nhân sự",
        "required": ["human resources", "recruitment", "talent acquisition"],
        "optional": ["payroll", "onboarding", "hris", "performance management"],
        "description": "HR Manager, Tuyển dụng...",
    },
    "BANKING": {
        "icon": "🏦", "display": "Ngân hàng",
        "required": ["banking", "credit analysis", "compliance"],
        "optional": ["loan", "kyc", "aml", "treasury"],
        "description": "Tín dụng, Giao dịch viên...",
    },
    "CONSULTANT": {
        "icon": "🧠", "display": "Tư vấn",
        "required": ["consulting", "management consultant", "strategy"],
        "optional": ["sap", "erp", "change management", "pmp"],
        "description": "Tư vấn quản lý, ERP...",
    },
    "DESIGNER": {
        "icon": "🎨", "display": "Thiết kế",
        "required": ["graphic design", "ui design", "figma"],
        "optional": ["photoshop", "illustrator", "ux research", "branding"],
        "description": "UI/UX, Đồ họa...",
    },
    "DIGITAL-MEDIA": {
        "icon": "📱", "display": "Truyền thông số",
        "required": ["digital marketing", "seo", "content creation"],
        "optional": ["google analytics", "facebook ads", "social media"],
        "description": "Digital Marketing, SEO...",
    },
    "PUBLIC-RELATIONS": {
        "icon": "📣", "display": "Quan hệ công chúng",
        "required": ["public relations", "media relations", "communications"],
        "optional": ["press release", "crisis management", "brand management"],
        "description": "PR Executive, Truyền thông...",
    },
    "AVIATION": {
        "icon": "✈️", "display": "Hàng không",
        "required": ["aviation", "flight", "airline"],
        "optional": ["pilot", "cabin crew", "icao", "safety"],
        "description": "Phi công, Tiếp viên...",
    },
    "AGRICULTURE": {
        "icon": "🌾", "display": "Nông nghiệp",
        "required": ["agriculture", "farming", "crop"],
        "optional": ["agronomy", "irrigation", "soil science", "livestock"],
        "description": "Kỹ sư nông nghiệp...",
    },
    "FITNESS": {
        "icon": "💪", "display": "Thể dục / Sức khỏe",
        "required": ["fitness", "personal training", "exercise"],
        "optional": ["nutrition", "yoga", "strength conditioning"],
        "description": "PT, HLV thể dục...",
    },
    "ARTS": {
        "icon": "🖼️", "display": "Nghệ thuật",
        "required": ["fine arts", "art", "creative"],
        "optional": ["photography", "illustration", "gallery", "digital art"],
        "description": "Họa sĩ, Nhiếp ảnh...",
    },
    "CHEF": {
        "icon": "👨‍🍳", "display": "Đầu bếp",
        "required": ["culinary", "kitchen management", "food preparation"],
        "optional": ["pastry", "haccp", "catering", "restaurant"],
        "description": "Đầu bếp, Quản lý bếp...",
    },
    "ADVOCATE": {
        "icon": "⚖️", "display": "Luật",
        "required": ["legal", "litigation", "law"],
        "optional": ["contract", "compliance", "corporate law", "arbitration"],
        "description": "Luật sư, Tư vấn pháp lý...",
    },
    "APPAREL": {
        "icon": "👗", "display": "Thời trang",
        "required": ["fashion", "apparel", "merchandising"],
        "optional": ["garment", "textile", "trend forecasting", "pattern making"],
        "description": "Thiết kế thời trang...",
    },
    "AUTOMOBILE": {
        "icon": "🚗", "display": "Ô tô",
        "required": ["automobile", "automotive", "mechanic"],
        "optional": ["engine", "transmission", "auto repair", "vehicle"],
        "description": "Kỹ thuật viên xe...",
    },
    "BPO": {
        "icon": "📞", "display": "BPO / Call Center",
        "required": ["customer service", "call center", "support"],
        "optional": ["crm", "data entry", "bpo", "quality assurance"],
        "description": "CSKH, Xử lý dữ liệu...",
    },
}

# ── Constants ──────────────────────────────────────────────────────────────────
HISTORY_FILE    = "Data/Processed/screening_history.json"
VECTORIZER_PATH = "Data/Processed/tfidf_vectorizer.pkl"
MODEL_PATH      = "Models/smartcv_model.pth"
CLASSES_PATH    = "Models/classes.npy"

PAGES = [
    "🏠  Trang chủ",
    "💼  Chọn vị trí",
    "⚙️  Cấu hình lọc",
    "📤  Upload CV",
    "📊  Kết quả & Biểu đồ",
    "📁  Lịch sử",
]

# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang tải mô hình AI...")
def load_model_artifacts():
    if not _torch_ok:
        return None, None, None
    if not all(os.path.exists(p) for p in [VECTORIZER_PATH, MODEL_PATH, CLASSES_PATH]):
        return None, None, None
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    input_size = len(vectorizer.get_feature_names_out())
    model = SmartCV_MLP(input_size=input_size, num_classes=len(classes))
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return vectorizer, model, classes


def predict_cv(cleaned_text, vectorizer, model, classes):
    vec = vectorizer.transform([cleaned_text]).toarray()
    tensor = torch.tensor(vec).float()
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(tensor), dim=1)[0]
    conf, pred = torch.max(probs, 0)
    all_probs = {classes[i]: float(probs[i]) * 100 for i in range(len(classes))}
    return classes[pred.item()], float(conf) * 100, all_probs


# ── History ────────────────────────────────────────────────────────────────────
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(records):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# ── Charts ─────────────────────────────────────────────────────────────────────
def gauge_chart(value):
    fig, ax = plt.subplots(figsize=(4, 2.8), subplot_kw=dict(aspect="equal"))
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.45, 1.3)
    ax.axis("off")

    theta_bg = np.linspace(np.pi, 0, 120)
    ax.plot(np.cos(theta_bg), np.sin(theta_bg), color="#ecf0f1", lw=22, solid_capstyle="round")

    for start, end, color in [(0.0, 0.4, "#e74c3c"), (0.4, 0.7, "#f39c12"), (0.7, 1.0, "#2ecc71")]:
        t = np.linspace(np.pi - start * np.pi, np.pi - end * np.pi, 60)
        ax.plot(np.cos(t), np.sin(t), color=color, lw=20, alpha=0.28, solid_capstyle="butt")

    v = max(0.0, min(value / 100, 1.0))
    color_val = "#e74c3c" if v < 0.4 else ("#f39c12" if v < 0.7 else "#2ecc71")
    t_val = np.linspace(np.pi, np.pi - v * np.pi, 120)
    ax.plot(np.cos(t_val), np.sin(t_val), color=color_val, lw=22, solid_capstyle="round")

    angle = np.pi - v * np.pi
    ax.annotate("", xy=(0.70 * np.cos(angle), 0.70 * np.sin(angle)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2.5))
    ax.add_patch(plt.Circle((0, 0), 0.07, color="#2c3e50", zorder=5))

    ax.text(0, -0.18, f"{value:.1f}%", ha="center", fontsize=20,
            fontweight="bold", color="#2c3e50")
    level = "Thấp" if v < 0.4 else ("Trung bình" if v < 0.7 else "Cao")
    ax.text(0, -0.33, f"Mức độ phù hợp: {level}", ha="center", fontsize=9, color=color_val)
    ax.text(-1.3, -0.18, "0%", fontsize=7, color="#95a5a6")
    ax.text(1.05, -0.18, "100%", fontsize=7, color="#95a5a6")
    plt.tight_layout()
    return fig


def skill_match_chart(resume_text, required_skills, optional_skills):
    all_skills = [(s, True) for s in required_skills] + [(s, False) for s in optional_skills]
    if not all_skills:
        return None

    labels, found_flags, bar_colors = [], [], []
    for skill, is_req in all_skills:
        hit = bool(check_keyword(resume_text, skill))
        labels.append(skill)
        found_flags.append(hit)
        bar_colors.append(
            ("#27ae60" if hit else "#e74c3c") if is_req
            else ("#2980b9" if hit else "#bdc3c7")
        )

    n = len(labels)
    fig, ax = plt.subplots(figsize=(7, max(2.5, n * 0.45)))
    ax.barh(range(n), [1] * n, color=bar_colors[::-1],
            height=0.65, edgecolor="white", linewidth=0.8)

    for i, (found, label) in enumerate(zip(found_flags[::-1], labels[::-1])):
        ax.text(0.5, i, "✓  Tìm thấy" if found else "✗  Không thấy",
                ha="center", va="center", fontsize=9,
                color="white" if found else "#555", fontweight="bold")
        ax.text(-0.02, i, label, ha="right", va="center", fontsize=8.5, color="#2c3e50")

    ax.set_xlim(-0.6, 1.15)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_title("Kiểm tra kỹ năng trong CV", fontsize=11, pad=10, fontweight="bold")

    patches = [
        mpatches.Patch(color="#27ae60", label="Bắt buộc — Có"),
        mpatches.Patch(color="#e74c3c", label="Bắt buộc — Thiếu"),
        mpatches.Patch(color="#2980b9", label="Ưu tiên — Có"),
        mpatches.Patch(color="#bdc3c7", label="Ưu tiên — Thiếu"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=7.5,
              ncol=2, framealpha=0.95, edgecolor="#ccc")
    plt.tight_layout()
    return fig


def top_categories_chart(all_probs, top_n=8, predicted=None):
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels = [x[0] for x in sorted_probs]
    values = [x[1] for x in sorted_probs]
    colors = [
        "#e74c3c" if lbl == predicted else ("#1f77b4" if i == 0 else "#aec7e8")
        for i, lbl in enumerate(labels)
    ]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1],
                   height=0.65, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Xác suất (%)", fontsize=9)
    ax.set_title(f"Top {top_n} ngành phù hợp nhất", fontsize=11, pad=8, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8.5)
    ax.set_xlim(0, max(values) * 1.28 if values else 1)
    plt.tight_layout()
    return fig


def logic_gate_figure(logic_result):
    gates = [
        ("AND", "Bắt buộc",  logic_result.get("mandatory_pass")),
        ("OR",  "Ưu tiên",   logic_result.get("optional_pass")),
        ("NOT", "Không cấm", logic_result.get("blacklist_pass")),
        ("XOR", "Độc quyền", logic_result.get("exclusive_pass")),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(9, 2.4))
    for ax, (name, label, val) in zip(axes, gates):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        if val is None:
            bg, fg, icon = "#f5f5f5", "#aaa", "—"
        elif val:
            bg, fg, icon = "#d5f5e3", "#27ae60", "✓"
        else:
            bg, fg, icon = "#fadbd8", "#e74c3c", "✗"
        ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9,
                                   facecolor=bg, edgecolor=fg, linewidth=2,
                                   transform=ax.transAxes, clip_on=False))
        ax.text(0.5, 0.68, name,  ha="center", va="center", fontsize=14,
                fontweight="bold", color=fg, transform=ax.transAxes)
        ax.text(0.5, 0.40, icon,  ha="center", va="center", fontsize=20,
                color=fg, transform=ax.transAxes)
        ax.text(0.5, 0.14, label, ha="center", va="center", fontsize=8,
                color="#555", transform=ax.transAxes)
    plt.tight_layout(pad=0.5)
    return fig


# ── Page config & CSS ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartCV — Sàng lọc CV tự động",
    page_icon="📄",
    layout="wide",
)

st.markdown("""
<style>
div[data-testid="stMetricValue"] { font-size: 1.3rem; }
.stButton > button { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("📄 SmartCV — Hệ thống Sàng lọc Hồ sơ Tự động")
st.caption("Kết hợp MLP Deep Learning + Logic Gates để đánh giá CV ứng viên")

if _import_errors:
    with st.expander("⚠️ Một số thư viện chưa cài", expanded=False):
        for err in _import_errors:
            st.error(err)
        st.code("pip install -r requirements.txt")

# ── Session state ──────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]
if "selected_job" not in st.session_state:
    st.session_state.selected_job = None
if "criteria" not in st.session_state:
    st.session_state.criteria = {
        "job_title": "", "required_skills": [],
        "optional_skills": [], "blacklist": [], "exclusive_pair": [],
    }
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_cv_name" not in st.session_state:
    st.session_state.last_cv_name = ""

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Điều hướng")
    cur_idx = PAGES.index(st.session_state.page) if st.session_state.page in PAGES else 0
    page = st.radio("Chọn module:", PAGES, index=cur_idx)
    st.session_state.page = page

    st.divider()
    c = st.session_state.criteria
    if c["job_title"]:
        st.success(f"**Vị trí:** {c['job_title']}")
        st.caption(
            f"AND: {len(c['required_skills'])} | "
            f"OR: {len(c['optional_skills'])} | "
            f"NOT: {len(c['blacklist'])}"
        )
    else:
        st.warning("Chưa chọn vị trí tuyển dụng")
    st.divider()
    st.caption("SmartCV v2.0 — HUIT Deep Learning Project")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0: TRANG CHỦ
# ══════════════════════════════════════════════════════════════════════════════
if page == PAGES[0]:
    history = load_history()
    passed = sum(1 for r in history if r.get("status") == "ĐẠT")
    c = st.session_state.criteria

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Vị trí đang tuyển", c["job_title"] or "Chưa chọn")
    col2.metric("Kỹ năng bắt buộc", len(c["required_skills"]))
    col3.metric("CV đã sàng lọc", len(history))
    col4.metric("Tỷ lệ ĐẠT", f"{passed/len(history)*100:.0f}%" if history else "—")

    st.divider()
    st.subheader("Quy trình hoạt động")
    steps = [
        ("1️⃣", "Chọn\nvị trí"), ("2️⃣", "Cấu hình\ntiêu chí"),
        ("3️⃣", "Upload\nCV PDF"), ("4️⃣", "Trích xuất\nvăn bản"),
        ("5️⃣", "Dịch ngôn\nngữ"), ("6️⃣", "Logic\nGates"),
        ("7️⃣", "MLP\nModel"), ("8️⃣", "Kết quả"),
    ]
    cols = st.columns(len(steps))
    for col, (num, label) in zip(cols, steps):
        col.markdown(
            f"<div style='text-align:center;padding:8px 4px;background:#f0f4ff;"
            f"border-radius:8px;font-size:11px;border:1px solid #dce8ff'>"
            f"<b>{num}</b><br>{label}</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    col_info, col_chart = st.columns([1.2, 1])
    with col_info:
        st.info(
            "**Hướng dẫn nhanh:**\n"
            "1. **💼 Chọn vị trí** → click card ngành nghề\n"
            "2. **⚙️ Cấu hình lọc** → điều chỉnh kỹ năng yêu cầu\n"
            "3. **📤 Upload CV** → tải file PDF lên\n"
            "4. **📊 Kết quả** → xem báo cáo chi tiết"
        )
        if st.button("🚀 Bắt đầu → Chọn vị trí", type="primary"):
            st.session_state.page = PAGES[1]
            st.rerun()

    with col_chart:
        if history:
            fig_pie, ax = plt.subplots(figsize=(3.5, 2.8))
            ax.pie(
                [passed, len(history) - passed],
                labels=["ĐẠT", "LOẠI"],
                colors=["#2ecc71", "#e74c3c"],
                autopct="%1.0f%%", startangle=90,
                textprops={"fontsize": 10},
                wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
            )
            ax.set_title("Tỷ lệ CV sàng lọc", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_pie)
            plt.close(fig_pie)
        else:
            st.info("Chưa có dữ liệu lịch sử.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: CHỌN VỊ TRÍ
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[1]:
    st.header("💼 Chọn vị trí tuyển dụng")

    options = {f"{v['icon']} {v['display']}": k for k, v in JOB_TEMPLATES.items()}
    cur_display = (
        f"{JOB_TEMPLATES[st.session_state.selected_job]['icon']} "
        f"{JOB_TEMPLATES[st.session_state.selected_job]['display']}"
        if st.session_state.selected_job else None
    )
    cur_idx = list(options.keys()).index(cur_display) if cur_display in options else 0

    selected_label = st.selectbox(
        "Chọn vị trí tuyển dụng",
        list(options.keys()),
        index=cur_idx,
    )

    job_key  = options[selected_label]
    job_info = JOB_TEMPLATES[job_key]

    st.info(
        f"**{job_info['icon']} {job_info['display']}** — {job_info['description']}  \n"
        f"Kỹ năng gợi ý bắt buộc: `{'`, `'.join(job_info['required'])}`  \n"
        f"Kỹ năng ưu tiên: `{'`, `'.join(job_info['optional'])}`"
    )

    if st.button("Xác nhận & Tiếp tục → Cấu hình lọc ⚙️", type="primary"):
        st.session_state.selected_job = job_key
        st.session_state.criteria = {
            "job_title":       job_info["display"],
            "required_skills": list(job_info["required"]),
            "optional_skills": list(job_info["optional"]),
            "blacklist":       [],
            "exclusive_pair":  [],
        }
        st.session_state.page = PAGES[2]
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: CẤU HÌNH LỌC
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[2]:
    st.header("⚙️ Cấu hình tiêu chí lọc CV")

    c = st.session_state.criteria
    if not c["job_title"]:
        st.warning("Chưa chọn vị trí. Hãy quay lại **💼 Chọn vị trí** trước.")
        if st.button("← Quay lại"):
            st.session_state.page = PAGES[1]
            st.rerun()
        st.stop()

    col_title, col_btn = st.columns([5, 1])
    col_title.subheader(f"📌 Vị trí: **{c['job_title']}**")
    if col_btn.button("🔄 Đổi vị trí"):
        st.session_state.page = PAGES[1]
        st.rerun()

    st.divider()

    def skill_editor(title, icon, bg_color, border_color, key):
        """Render a skill list with add/remove controls."""
        skills = st.session_state.criteria[key]
        st.markdown(f"**{icon} {title}**")

        if skills:
            tags_html = " ".join(
                f"<span style='display:inline-block;background:{bg_color}18;"
                f"border:1px solid {border_color};border-radius:16px;"
                f"padding:3px 12px;margin:3px;font-size:12px;color:{border_color}'>"
                f"{s}</span>"
                for s in skills
            )
            st.markdown(tags_html, unsafe_allow_html=True)
            st.write("")
            rm_col, btn_col = st.columns([5, 1])
            to_rm = rm_col.selectbox(
                "rm", ["— chọn để xóa —"] + skills,
                key=f"rm_sel_{key}", label_visibility="collapsed",
            )
            if btn_col.button("✕ Xóa", key=f"rm_btn_{key}", use_container_width=True):
                if to_rm != "— chọn để xóa —":
                    st.session_state.criteria[key].remove(to_rm)
                    st.rerun()
        else:
            st.caption("Chưa có mục nào.")

        add_col, btn_col2 = st.columns([5, 1])
        new_val = add_col.text_input(
            "add", key=f"inp_{key}",
            placeholder="Nhập kỹ năng rồi bấm ➕ Thêm...",
            label_visibility="collapsed",
        )
        if btn_col2.button("➕ Thêm", key=f"add_btn_{key}", use_container_width=True):
            s = new_val.strip().lower()
            if s and s not in st.session_state.criteria[key]:
                st.session_state.criteria[key].append(s)
                st.rerun()
            elif s:
                st.toast(f"'{s}' đã có trong danh sách!", icon="⚠️")

    col_left, col_right = st.columns(2)

    with col_left:
        with st.container(border=True):
            skill_editor(
                "AND Gate — Kỹ năng BẮT BUỘC",
                "🔴", "#e74c3c", "#e74c3c", "required_skills",
            )
        st.write("")
        with st.container(border=True):
            skill_editor(
                "NOT Gate — Từ khóa CẤM",
                "⛔", "#8e44ad", "#8e44ad", "blacklist",
            )

    with col_right:
        with st.container(border=True):
            skill_editor(
                "OR Gate — Kỹ năng ƯU TIÊN",
                "🟡", "#f39c12", "#e67e22", "optional_skills",
            )
        st.write("")
        with st.container(border=True):
            st.markdown("**🔵 XOR Gate — Cặp kỹ năng ĐỘC QUYỀN**")
            st.caption("Chấp nhận CV có đúng MỘT trong hai kỹ năng.")
            xa_col, xb_col = st.columns(2)
            xa = xa_col.text_input(
                "Kỹ năng A", key="xor_a",
                value=c["exclusive_pair"][0] if len(c["exclusive_pair"]) > 0 else "",
            )
            xb = xb_col.text_input(
                "Kỹ năng B", key="xor_b",
                value=c["exclusive_pair"][1] if len(c["exclusive_pair"]) > 1 else "",
            )
            if st.button("💾 Cập nhật XOR", key="save_xor"):
                st.session_state.criteria["exclusive_pair"] = [
                    s for s in [xa.strip(), xb.strip()] if s
                ]
                st.toast("Đã cập nhật XOR Gate!", icon="✅")

    st.divider()
    col_summary, col_next = st.columns([2, 1])
    c = st.session_state.criteria
    with col_summary:
        st.success(
            f"**Tổng kết:**  \n"
            f"🔴 AND: **{len(c['required_skills'])}** kỹ năng bắt buộc  |  "
            f"🟡 OR: **{len(c['optional_skills'])}** kỹ năng ưu tiên  |  "
            f"⛔ NOT: **{len(c['blacklist'])}** từ cấm  |  "
            f"🔵 XOR: **{'Có' if c['exclusive_pair'] else 'Không'}**"
        )
    with col_next:
        if st.button("Tiếp tục → Upload CV 📤", type="primary", use_container_width=True):
            st.session_state.page = PAGES[3]
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: UPLOAD CV
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[3]:
    st.header("📤 Upload CV — Phân tích Hồ sơ")

    missing_mods = []
    if not _extractor_ok: missing_mods.append("Extractor")
    if not _cleaner_ok:   missing_mods.append("Cleaner")
    if not _gates_ok:     missing_mods.append("Logic Gates")
    if not _torch_ok:     missing_mods.append("PyTorch")
    if missing_mods:
        st.error(f"Thiếu module: {', '.join(missing_mods)}")
        st.code("pip install -r requirements.txt")
        st.stop()

    vectorizer, model, classes = load_model_artifacts()
    if vectorizer is None:
        st.error("Chưa có model! Hãy chạy `python main.py` → `python train.py` trước.")
        st.stop()

    c = st.session_state.criteria
    if not c["job_title"]:
        st.warning("Chưa cấu hình tiêu chí.")
        if st.button("← Chọn vị trí"):
            st.session_state.page = PAGES[1]
            st.rerun()
        st.stop()

    st.info(
        f"**Vị trí:** {c['job_title']}  |  "
        f"Bắt buộc: {len(c['required_skills'])} kỹ năng  |  "
        f"Ưu tiên: {len(c['optional_skills'])} kỹ năng"
    )

    uploaded = st.file_uploader("Tải lên file CV (PDF)", type=["pdf"])
    if uploaded is None:
        st.info("Vui lòng chọn file PDF để bắt đầu phân tích.")
        st.stop()

    st.success(f"Đã nhận file: **{uploaded.name}**")

    if st.button("🚀 Phân tích CV", type="primary"):
        uploaded.seek(0)
        file_bytes = uploaded.read()
        if len(file_bytes) == 0:
            st.error("File PDF rỗng.")
            st.stop()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        debug = {}
        raw_text = ""

        with st.spinner("Đang trích xuất văn bản từ PDF..."):
            try:
                t1 = extract_text_with_pdfplumber(tmp_path)
                debug["pdfplumber"] = f"{len(t1)} ký tự"
            except Exception as e:
                t1 = ""
                debug["pdfplumber"] = f"LỖI: {e}"
            try:
                t2 = extract_text_with_pypdf2(tmp_path)
                debug["pypdf2"] = f"{len(t2)} ký tự"
            except Exception as e:
                t2 = ""
                debug["pypdf2"] = f"LỖI: {e}"

            raw_text = t1 if len(t1) >= len(t2) else t2

            if len(raw_text.strip()) < 50:
                try:
                    imgs = _pdf_to_images_fitz(tmp_path)
                    import easyocr
                    reader = easyocr.Reader(["en"], verbose=False)
                    ocr_parts = []
                    for i, img in enumerate(imgs):
                        tmp_img = f"temp_ocr_{i}.jpg"
                        img.save(tmp_img, "JPEG")
                        result = reader.readtext(tmp_img, detail=0)
                        try:
                            os.remove(tmp_img)
                        except OSError:
                            pass
                        ocr_parts.append(" ".join(result))
                    raw_text = "\n".join(ocr_parts)
                    debug["ocr"] = f"{len(raw_text)} ký tự"
                except Exception as e:
                    debug["ocr"] = f"LỖI: {e}"

        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        with st.expander("🔍 Chi tiết trích xuất", expanded=False):
            for k, v in debug.items():
                icon = "❌" if "LỖI" in str(v) else "✅"
                st.write(f"{icon} **{k}**: `{v}`")
            if raw_text.strip():
                st.text_area("Raw text (300 ký tự đầu)", raw_text[:300], height=100)

        if not raw_text.strip():
            st.error("Không trích xuất được văn bản.")
            st.stop()

        with st.spinner("Kiểm tra ngôn ngữ và dịch (nếu cần)..."):
            raw_text = translate_if_needed(raw_text)

        with st.spinner("Làm sạch văn bản..."):
            cleaned = clean_text(raw_text)

        with st.spinner("Kiểm tra Logic Gates..."):
            logic_result = evaluate_candidate(
                cleaned,
                required_skills=c["required_skills"],
                optional_skills=c["optional_skills"],
                blacklist=c["blacklist"],
                exclusive_pair=c["exclusive_pair"] if len(c["exclusive_pair"]) == 2 else None,
            )

        # Luôn chạy MLP bất kể Logic Gate — tránh bỏ sót CV tốt do HR cấu hình sai
        with st.spinner("Phân loại bằng MLP..."):
            category, confidence, all_probs = predict_cv(cleaned, vectorizer, model, classes)

        if not logic_result["overall_pass"]:
            reasons = []
            if not logic_result["mandatory_pass"]:
                missing = [s for s in c["required_skills"]
                           if not check_keyword(cleaned, s)]
                reasons.append(f"Thiếu kỹ năng: {', '.join(missing)}")
            if not logic_result["blacklist_pass"]:
                reasons.append(f"Chứa từ cấm: {', '.join(c['blacklist'])}")
            result = {
                "status": "LOẠI", "reason": " | ".join(reasons),
                "category": category, "confidence": confidence,
                "text": cleaned, "logic": logic_result, "all_probs": all_probs,
                "logic_failed": True,
            }
        else:
            result = {
                "status": "ĐẠT", "reason": "Đáp ứng tất cả tiêu chí",
                "category": category, "confidence": confidence,
                "text": cleaned, "logic": logic_result, "all_probs": all_probs,
                "logic_failed": False,
            }

        st.session_state.last_result  = result
        st.session_state.last_cv_name = uploaded.name

        history = load_history()
        history.append({
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename":   uploaded.name,
            "job_title":  c["job_title"],
            "status":     result["status"],
            "category":   result["category"],
            "confidence": round(result["confidence"], 2),
            "reason":     result["reason"],
        })
        save_history(history)
        st.rerun()

    if st.session_state.last_result and st.session_state.last_cv_name:
        r = st.session_state.last_result
        st.divider()
        if r["status"] == "ĐẠT":
            st.success(f"✅ **ĐẠT** — {r['category']} ({r['confidence']:.1f}%)")
        else:
            st.error(f"❌ **LOẠI** — {r['reason']}")
            if r.get("category") and r.get("confidence", 0) > 30:
                st.warning(
                    f"⚠️ **Lưu ý:** AI vẫn nhận diện CV này là **{r['category']}** "
                    f"(độ tin cậy {r['confidence']:.1f}%). "
                    f"Có thể tiêu chí kỹ năng bắt buộc chưa phù hợp — HR nên xem xét lại."
                )
        if st.button("Xem chi tiết → Kết quả & Biểu đồ 📊", type="secondary"):
            st.session_state.page = PAGES[4]
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: KẾT QUẢ & BIỂU ĐỒ
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[4]:
    st.header("📊 Kết quả & Biểu đồ Phân tích")

    r = st.session_state.last_result
    if r is None:
        st.info("Chưa có kết quả. Hãy upload CV ở trang trước.")
        st.stop()

    cv_name = st.session_state.last_cv_name
    c = st.session_state.criteria

    col1, col2, col3, col4 = st.columns(4)
    status_icon = "🟢" if r["status"] == "ĐẠT" else "🔴"
    col1.metric("Kết quả",        f"{status_icon} {r['status']}")
    col2.metric("Ngành dự đoán",  r["category"] or "—")
    col3.metric("Độ tin cậy AI",  f"{r['confidence']:.1f}%" if r["confidence"] else "—")
    col4.metric("Vị trí tuyển",   c["job_title"])
    st.caption(f"File: **{cv_name}**")

    st.divider()

    # Logic Gates visual
    st.subheader("Kết quả Logic Gates")
    fig_gates = logic_gate_figure(r.get("logic", {}))
    st.pyplot(fig_gates)
    plt.close(fig_gates)

    st.divider()

    # Gauge + Top categories
    if r["status"] == "ĐẠT":
        col_g, col_bar = st.columns([1.1, 2])
        with col_g:
            st.subheader("Độ tin cậy AI")
            fig_g = gauge_chart(r["confidence"])
            st.pyplot(fig_g)
            plt.close(fig_g)
        with col_bar:
            st.subheader("Top ngành phù hợp nhất")
            if r["all_probs"]:
                fig_bar = top_categories_chart(r["all_probs"], predicted=r["category"])
                st.pyplot(fig_bar)
                plt.close(fig_bar)
        st.divider()

    # Skill match chart
    all_req = c["required_skills"]
    all_opt = c["optional_skills"]
    if (all_req or all_opt) and _gates_ok:
        st.subheader("Phân tích kỹ năng trong CV")
        fig_skill = skill_match_chart(r["text"], all_req, all_opt)
        if fig_skill:
            st.pyplot(fig_skill)
            plt.close(fig_skill)
        st.divider()

    with st.expander("Xem văn bản CV sau tiền xử lý NLP"):
        st.text_area("Cleaned text", value=r["text"], height=200, disabled=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: LỊCH SỬ
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[5]:
    st.header("📁 Lịch sử Sàng lọc Hồ sơ")

    history = load_history()
    if not history:
        st.info("Chưa có lịch sử. Hãy phân tích ít nhất một CV.")
        st.stop()

    df_hist = pd.DataFrame(history)
    total  = len(df_hist)
    passed = (df_hist["status"] == "ĐẠT").sum()
    failed = total - passed

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng CV",   total)
    col2.metric("ĐẠT",       passed)
    col3.metric("LOẠI",      failed)
    col4.metric("Tỷ lệ ĐẠT", f"{passed/total*100:.0f}%" if total else "—")

    col_pie, col_bar = st.columns(2)
    with col_pie:
        fig_pie, ax = plt.subplots(figsize=(4, 3.2))
        ax.pie(
            [passed, failed],
            labels=["ĐẠT", "LOẠI"],
            colors=["#2ecc71", "#e74c3c"],
            autopct="%1.0f%%", startangle=90,
            textprops={"fontsize": 10},
            wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
        )
        ax.set_title("Tỷ lệ kết quả sàng lọc", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig_pie)
        plt.close(fig_pie)

    with col_bar:
        cat_counts = df_hist[df_hist["category"].notna()]["category"].value_counts().head(8)
        if not cat_counts.empty:
            fig_cat, ax2 = plt.subplots(figsize=(5.5, 3.5))
            bars = ax2.barh(
                cat_counts.index[::-1], cat_counts.values[::-1],
                color="#1f77b4", height=0.6,
            )
            ax2.set_xlabel("Số lượng CV")
            ax2.set_title("Phân bố ngành nghề (CV ĐẠT)", fontsize=11)
            ax2.spines[["top", "right"]].set_visible(False)
            ax2.grid(axis="x", alpha=0.3, linestyle="--")
            for bar, val in zip(bars, cat_counts.values[::-1]):
                ax2.text(bar.get_width() + 0.05,
                         bar.get_y() + bar.get_height() / 2,
                         str(val), va="center", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_cat)
            plt.close(fig_cat)

    st.divider()
    st.subheader("Chi tiết từng hồ sơ")

    display_df = df_hist[["timestamp", "filename", "job_title",
                           "status", "category", "confidence"]].copy()
    display_df.columns = ["Thời gian", "File CV", "Vị trí", "Kết quả", "Ngành", "Tin cậy (%)"]
    display_df = display_df.sort_values("Thời gian", ascending=False).reset_index(drop=True)

    st.dataframe(
        display_df.style.map(
            lambda v: (
                "color:#27ae60;font-weight:bold" if v == "ĐẠT"
                else "color:#e74c3c;font-weight:bold" if v == "LOẠI"
                else ""
            ),
            subset=["Kết quả"],
        ),
        use_container_width=True,
        height=350,
    )

    col_del, col_dl = st.columns([1, 1])
    if col_del.button("🗑️ Xóa toàn bộ lịch sử", type="secondary"):
        save_history([])
        st.success("Đã xóa lịch sử.")
        st.rerun()

    csv_data = display_df.to_csv(index=False).encode("utf-8-sig")
    col_dl.download_button(
        "⬇️ Tải xuống CSV", data=csv_data,
        file_name="smartcv_history.csv", mime="text/csv",
    )
