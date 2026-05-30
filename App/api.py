"""
SmartCV Web API — FastAPI backend
Kiến trúc 4 tầng: OCR → BERT Encoder → MLP Head → Logic Gates
Chạy: python run_app.py  →  http://localhost:8000
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import uuid
import tempfile
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import torch
from SRC.model_bert import SmartCV_BERT
from SRC.logic_gates import evaluate_candidate, check_keyword

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE      = "Data/Processed/screening_history.json"
CURRENT_JOB_FILE  = "Data/Processed/current_job.json"
BERT_MODEL_PATH   = "Models/bert_classifier.pth"
BERT_CONFIG_PATH  = "Models/bert_config.json"
CKPT_LOG          = "Models/checkpoints/log.json"

# ── Job templates (6 ngành) ────────────────────────────────────────────────────
JOB_TEMPLATES = {
    "INFORMATION-TECHNOLOGY": {
        "icon": "💻", "display": "Công nghệ thông tin",
        "required": ["python", "sql", "software development"],
        "optional": ["machine learning", "docker", "aws", "javascript", "linux"],
        "description": "Dev, Data Science, DevOps, IT...",
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
    "DIGITAL-MEDIA": {
        "icon": "📱", "display": "Truyền thông số / Marketing",
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
    "CONSULTANT": {
        "icon": "🧠", "display": "Tư vấn",
        "required": ["consulting", "management consultant", "strategy"],
        "optional": ["sap", "erp", "change management", "pmp"],
        "description": "Tư vấn quản lý, ERP...",
    },
}

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="SmartCV API")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ── Model cache ────────────────────────────────────────────────────────────────
_bert_cache: dict = {}

def load_bert_artifacts():
    """Load BERT classification head + config. Returns (model, config) or (None, None)."""
    if "loaded" in _bert_cache:
        return _bert_cache["model"], _bert_cache["config"]
    if not (os.path.exists(BERT_MODEL_PATH) and os.path.exists(BERT_CONFIG_PATH)):
        return None, None
    with open(BERT_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    bert_model = SmartCV_BERT(embed_dim=config["embed_dim"], num_classes=config["num_classes"])
    try:
        bert_model.load_state_dict(
            torch.load(BERT_MODEL_PATH, map_location="cpu", weights_only=True)
        )
    except TypeError:
        bert_model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location="cpu"))
    bert_model.eval()
    _bert_cache.update({"model": bert_model, "config": config, "loaded": True})
    return bert_model, config

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(records: list):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def load_current_job() -> Optional[dict]:
    if os.path.exists(CURRENT_JOB_FILE):
        with open(CURRENT_JOB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_current_job(job: dict):
    os.makedirs(os.path.dirname(CURRENT_JOB_FILE), exist_ok=True)
    with open(CURRENT_JOB_FILE, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)

def _predict_bert(cleaned: str, bert_model, config) -> tuple:
    """
    Encode CV bằng BERT → MLP phân loại ngành.
    Trả về (category, confidence, all_probs, cv_embedding).
    cv_embedding được dùng lại bởi Logic Gates (BERT semantic skill check).
    """
    from SRC.Processing.bert_encoder import encode_single
    classes = config["classes"]
    emb     = encode_single(cleaned)          # shape (1, 384) — dùng lại cho Logic Gates
    tensor  = torch.tensor(emb, dtype=torch.float32)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(bert_model(tensor), dim=1)[0]
    conf, pred = torch.max(probs, 0)
    all_probs  = {classes[i]: float(probs[i]) * 100 for i in range(len(classes))}
    return classes[pred.item()], float(conf) * 100, all_probs, emb

def _compute_stats(history: list) -> dict:
    total  = len(history)
    passed = sum(1 for r in history if r.get("status") == "ĐẠT")
    by_cat: dict = {}
    for r in history:
        cat = r.get("category") or "Unknown"
        by_cat[cat] = by_cat.get(cat, 0) + 1
    return {
        "total":       total,
        "passed":      passed,
        "failed":      total - passed,
        "pass_rate":   round(passed / total * 100, 1) if total else 0,
        "by_category": by_cat,
    }

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    history     = load_history()
    current_job = load_current_job()
    stats       = _compute_stats(history)
    bert_model, bert_config = load_bert_artifacts()

    return templates.TemplateResponse(request, "dashboard.html", {
        "stats":           stats,
        "current_job":     current_job,
        "model_ready":     bert_model is not None,
        "recent_activity": list(reversed(history))[:5],
        "active_page":     "dashboard",
    })


@app.get("/jobs/new", response_class=HTMLResponse)
async def jobs_new(request: Request):
    current_job = load_current_job()
    return templates.TemplateResponse(request, "jobs/new.html", {
        "job_templates": JOB_TEMPLATES,
        "current_job":   current_job,
        "active_page":   "jobs",
    })


@app.post("/jobs")
async def create_job(request: Request):
    data = await request.json()
    job_key = data.get("job_key", "")
    tpl     = JOB_TEMPLATES.get(job_key, {})
    job = {
        "job_key":         job_key,
        "job_title":       tpl.get("display", job_key),
        "required_skills": data.get("required_skills", []),
        "optional_skills": data.get("optional_skills", []),
        "blacklist":       data.get("blacklist", []),
        "exclusive_pair":  data.get("exclusive_pair", []),
    }
    save_current_job(job)
    return {"success": True}


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    current_job = load_current_job()
    return templates.TemplateResponse(request, "upload.html", {
        "current_job": current_job,
        "active_page": "upload",
    })


@app.post("/candidates/upload")
async def upload_cv(files: List[UploadFile] = File(...)):
    current_job = load_current_job()
    if not current_job:
        return JSONResponse({"error": "Chưa cấu hình job"}, status_code=400)

    bert_model, bert_config = load_bert_artifacts()
    if bert_model is None:
        return JSONResponse(
            {"error": "Chưa có model. Chạy train_bert.py trước."},
            status_code=400,
        )

    try:
        from SRC.Processing.extractor import (
            extract_text_with_pdfplumber,
            extract_text_with_pypdf2,
            _pdf_to_images_fitz,
            _get_ocr_reader,
            _preprocess_image,
        )
        from SRC.Processing.cleaner import clean_text, translate_if_needed
    except Exception as e:
        return JSONResponse({"error": f"Import lỗi: {e}"}, status_code=500)

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    results = []
    history = load_history()

    for file in files:
        try:
            fname    = file.filename or ""
            ext      = os.path.splitext(fname)[1].lower()
            is_image = ext in IMAGE_EXTS
            suffix   = ext if ext else ".pdf"

            file_bytes = await file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            # ── Tầng 1: Trích xuất văn bản ──────────────────────────────────
            if is_image:
                from PIL import Image
                img      = Image.open(tmp_path)
                img_pre  = _preprocess_image(img)
                pre_path = tmp_path + "_pre.jpg"
                img_pre.save(pre_path, "JPEG", quality=95)
                reader     = _get_ocr_reader()
                lines_orig = reader.readtext(tmp_path, detail=0)
                lines_pre  = reader.readtext(pre_path, detail=0)
                try:
                    os.unlink(pre_path)
                except OSError:
                    pass
                seen, merged = set(), []
                for t in lines_orig + lines_pre:
                    t_norm = t.strip().lower()
                    if t_norm and t_norm not in seen:
                        seen.add(t_norm)
                        merged.append(t)
                raw_text = " ".join(merged)
            else:
                try:
                    t1 = extract_text_with_pdfplumber(tmp_path)
                except Exception:
                    t1 = ""
                try:
                    t2 = extract_text_with_pypdf2(tmp_path)
                except Exception:
                    t2 = ""
                raw_text = t1 if len(t1) >= len(t2) else t2

                if len(raw_text.strip()) < 50:
                    try:
                        imgs    = _pdf_to_images_fitz(tmp_path)
                        reader  = _get_ocr_reader()
                        ocr_out = []
                        for i, img in enumerate(imgs):
                            img = _preprocess_image(img)
                            tmp_img = f"temp_ocr_{i}.jpg"
                            img.save(tmp_img, "JPEG", quality=95)
                            ocr_out.append(" ".join(reader.readtext(tmp_img, detail=0)))
                            try:
                                os.remove(tmp_img)
                            except OSError:
                                pass
                        raw_text = "\n".join(ocr_out)
                    except Exception:
                        pass

            try:
                os.unlink(tmp_path)
            except OSError:
                pass

            # Phát hiện ngôn ngữ
            lang = "en"
            try:
                from SRC.Processing.cleaner import _has_vietnamese
                if _has_vietnamese(raw_text):
                    lang = "vi"
                else:
                    from langdetect import detect
                    lang = detect(raw_text) if raw_text.strip() else "en"
            except Exception:
                pass

            # Dịch + làm sạch
            raw_text = translate_if_needed(raw_text)
            cleaned  = clean_text(raw_text)

            # ── Tầng 2+3: BERT Encoder → MLP Head ───────────────────────────
            # Encode trước để cv_embedding dùng lại cho Logic Gates (tránh encode 2 lần)
            category, confidence, all_probs, cv_embedding = _predict_bert(
                cleaned, bert_model, bert_config
            )

            # ── Tầng 4: Logic Gates (hybrid: keyword + BERT semantic) ─────────
            c            = current_job
            xpair        = c.get("exclusive_pair", [])
            logic_result = evaluate_candidate(
                cleaned,
                required_skills=c.get("required_skills", []),
                optional_skills=c.get("optional_skills", []),
                blacklist=c.get("blacklist", []),
                exclusive_pair=xpair if len(xpair) == 2 else None,
                cv_embedding=cv_embedding,   # BERT embedding → smart skill detection
            )

            # Kết quả ĐẠT/LOẠI
            if logic_result["overall_pass"]:
                status = "ĐẠT"
                reason = "Đáp ứng tất cả tiêu chí"
            else:
                reasons = []
                if not logic_result.get("mandatory_pass"):
                    missing = [s for s in c.get("required_skills", [])
                               if not check_keyword(cleaned, s)]
                    if missing:
                        reasons.append(f"Thiếu kỹ năng: {', '.join(missing)}")
                if not logic_result.get("blacklist_pass"):
                    reasons.append("Chứa từ khóa bị cấm")
                status = "LOẠI"
                reason = " | ".join(reasons) if reasons else "Không đạt tiêu chí"

            record_id = str(uuid.uuid4())
            record = {
                "id":              record_id,
                "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename":        file.filename,
                "job_title":       c.get("job_title", ""),
                "job_key":         c.get("job_key", ""),
                "status":          status,
                "category":        category,
                "confidence":      round(confidence, 2),
                "reason":          reason,
                "all_probs":       all_probs,
                "logic_result":    logic_result,
                "cleaned_text":    cleaned[:3000],
                "required_skills": c.get("required_skills", []),
                "optional_skills": c.get("optional_skills", []),
                "lang":            lang,
                "model_used":      "BERT",
            }
            history.append(record)

            logic_checks = {
                "mandatory": logic_result.get("mandatory_pass"),
                "optional":  logic_result.get("optional_pass"),
                "blacklist": logic_result.get("blacklist_pass"),
                "exclusive": logic_result.get("exclusive_pass"),
            }
            checked      = [v for v in logic_checks.values() if v is not None]
            passed_count = sum(1 for v in checked if v)

            results.append({
                "id":           record_id,
                "filename":     file.filename,
                "status":       status,
                "category":     category,
                "confidence":   round(confidence, 2),
                "reason":       reason,
                "lang":         lang,
                "logic_checks": logic_checks,
                "logic_score":  f"{passed_count}/{len(checked)}",
                "model_used":   "BERT",
            })

        except Exception as e:
            results.append({
                "filename":   file.filename,
                "status":     "LỖI",
                "error":      str(e),
                "confidence": 0,
            })

    save_history(history)
    return {"results": results}


@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    history = load_history()
    return templates.TemplateResponse(request, "results.html", {
        "history":     list(reversed(history)),
        "active_page": "results",
    })


@app.get("/candidates/{candidate_id}", response_class=HTMLResponse)
async def candidate_detail(request: Request, candidate_id: str):
    history   = load_history()
    candidate = next((r for r in history if r.get("id") == candidate_id), None)
    if not candidate:
        return HTMLResponse("<h1>Không tìm thấy ứng viên</h1>", status_code=404)
    tpl = JOB_TEMPLATES.get(candidate.get("job_key", ""), {})
    return templates.TemplateResponse(request, "candidate_detail.html", {
        "candidate":    candidate,
        "job_template": tpl,
        "active_page":  "results",
    })


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    history = load_history()
    stats   = _compute_stats(history)
    return templates.TemplateResponse(request, "history.html", {
        "history":     list(reversed(history)),
        "stats":       stats,
        "active_page": "history",
    })


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    bert_model, bert_config = load_bert_artifacts()
    current_job = load_current_job()

    checkpoints = []
    if os.path.exists(CKPT_LOG):
        with open(CKPT_LOG) as f:
            log = json.load(f)
        checkpoints = sorted(log, key=lambda x: x.get("val_f1", 0), reverse=True)[:5]

    return templates.TemplateResponse(request, "settings.html", {
        "bert_ready":   bert_model is not None,
        "bert_config":  bert_config or {},
        "checkpoints":  checkpoints,
        "current_job":  current_job,
        "active_page":  "settings",
    })


@app.post("/api/clear-history")
async def clear_history():
    save_history([])
    return {"success": True}


@app.get("/api/stats")
async def api_stats():
    return _compute_stats(load_history())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("App.api:app", host="0.0.0.0", port=8000, reload=True)
