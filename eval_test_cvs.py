"""
Đánh giá hệ thống 4 tầng trên thư mục Test_CVs.
Pipeline: OCR/PDF -> BERT Encoder -> MLP Head -> kết quả
Output: visual report + CSV
"""
import os
import json
import csv
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from SRC.Processing.cleaner import clean_text, translate_if_needed
from SRC.Processing.extractor import extract_text, _get_ocr_reader
from SRC.model_bert import SmartCV_BERT
from SRC.Processing.bert_encoder import encode_texts

# ── Config ────────────────────────────────────────────────────────────────────
TEST_CVS_DIR     = "Data/Test_CVs"
OUTPUT_DIR       = "Models/test_evaluation"
BERT_MODEL_PATH  = "Models/bert_classifier.pth"
BERT_CONFIG_PATH = "Models/bert_config.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = [
    "BUSINESS-DEVELOPMENT", "CONSULTANT", "DIGITAL-MEDIA",
    "INFORMATION-TECHNOLOGY", "PUBLIC-RELATIONS", "SALES"
]

# Ground truth theo thư mục/tên file
# 6_nganh_chinh: subfolder name = ground truth label (auto-detect)
GROUND_TRUTH = {
    "ocr_test": {
        "Information_Technology.jpg":          "INFORMATION-TECHNOLOGY",
        "Information_Technology_software.jpg": "INFORMATION-TECHNOLOGY",
        "Software_Enginee.jpg":                "INFORMATION-TECHNOLOGY",
    },
    "tieng_viet": {
        "CV NGUYENMINHDAT.pdf": "PUBLIC-RELATIONS",
    },
    "ngoai_he_thong": {},   # ngoài hệ thống → ground truth = UNKNOWN
}

VALID_CLASSES = set(CLASSES)  # subfolder names dùng làm ground truth tự động

# ── Load model ────────────────────────────────────────────────────────────────
print("=" * 70)
print("LOADING MODEL (BERT + MLP)...")
print("=" * 70)

if not (os.path.exists(BERT_MODEL_PATH) and os.path.exists(BERT_CONFIG_PATH)):
    raise FileNotFoundError("Chua co model. Chay train_bert.py truoc!")

with open(BERT_CONFIG_PATH, "r", encoding="utf-8") as f:
    bert_config = json.load(f)

model = SmartCV_BERT(embed_dim=bert_config["embed_dim"], num_classes=bert_config["num_classes"])
model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location="cpu"))
model.eval()
print("[OK] BERT + MLP loaded  |  embed_dim=" + str(bert_config["embed_dim"]) +
      "  num_classes=" + str(bert_config["num_classes"]))

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EVALUATING TEST_CVS...")
print("=" * 70)

results = []

for folder in sorted(os.listdir(TEST_CVS_DIR)):
    folder_path = os.path.join(TEST_CVS_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    print("\n--- " + folder + "/ ---")

    # 6_nganh_chinh: duyệt qua subfolder, tên subfolder = ground truth
    if folder == "6_nganh_chinh":
        for sub in sorted(os.listdir(folder_path)):
            sub_path = os.path.join(folder_path, sub)
            if not os.path.isdir(sub_path):
                continue
            ground_truth = sub if sub in VALID_CLASSES else "UNKNOWN"
            print("  [" + sub + "]")
            for filename in sorted(os.listdir(sub_path)):
                file_path = os.path.join(sub_path, filename)
                if not os.path.isfile(file_path):
                    continue
                try:
                    if filename.lower().endswith(".pdf"):
                        raw_text = extract_text(file_path)
                    elif filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        reader   = _get_ocr_reader()
                        raw_text = " ".join(reader.readtext(file_path, detail=0))
                    else:
                        print("    [skip] " + filename)
                        continue
                    if not raw_text or len(raw_text.strip()) < 50:
                        print("    [skip] " + filename + ": extraction failed")
                        continue
                    text      = clean_text(translate_if_needed(raw_text))
                    embedding = encode_texts([text])
                    with torch.no_grad():
                        probs = torch.nn.functional.softmax(
                            model(torch.tensor(embedding, dtype=torch.float32)), dim=1
                        )[0].numpy()
                    pred_idx   = int(np.argmax(probs))
                    prediction = CLASSES[pred_idx]
                    confidence = float(probs[pred_idx]) * 100
                    correct    = (prediction == ground_truth)
                    results.append({
                        "folder":       folder + "/" + sub,
                        "filename":     filename,
                        "ground_truth": ground_truth,
                        "prediction":   prediction,
                        "confidence":   round(confidence, 2),
                        "correct":      bool(correct),
                    })
                    status = "[OK]" if correct else "[X] "
                    print("    " + status + " " + filename + "  ->  " + prediction + " (" + str(round(confidence,1)) + "%)")
                except Exception as e:
                    print("    [ERR] " + filename + ": " + str(e)[:60])
        continue  # skip generic processing below for this folder

    gt_map = GROUND_TRUTH.get(folder, {})

    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue

        ground_truth = gt_map.get(filename, "UNKNOWN")

        try:
            # Tầng 1: Trích xuất văn bản
            if filename.lower().endswith(".pdf"):
                raw_text = extract_text(file_path)
            elif filename.lower().endswith((".jpg", ".jpeg", ".png")):
                reader   = _get_ocr_reader()
                raw_text = " ".join(reader.readtext(file_path, detail=0))
            else:
                print("  [skip] " + filename + ": unsupported format")
                continue

            if not raw_text or len(raw_text.strip()) < 50:
                print("  [skip] " + filename + ": extraction failed")
                continue

            # Tiền xử lý
            text = clean_text(translate_if_needed(raw_text))

            # Tầng 2+3: BERT Encoder -> MLP Head
            embedding = encode_texts([text])
            with torch.no_grad():
                probs = torch.nn.functional.softmax(
                    model(torch.tensor(embedding, dtype=torch.float32)), dim=1
                )[0].numpy()

            pred_idx   = int(np.argmax(probs))
            prediction = CLASSES[pred_idx]
            confidence = float(probs[pred_idx]) * 100
            correct    = (prediction == ground_truth)

            results.append({
                "folder":       folder,
                "filename":     filename,
                "ground_truth": ground_truth,
                "prediction":   prediction,
                "confidence":   round(confidence, 2),
                "correct":      bool(correct),
            })

            status = "[OK]" if correct else "[X] "
            print("  " + status + " " + filename)
            print("       Ground: " + ground_truth + "  |  Pred: " + prediction +
                  " (" + str(round(confidence, 1)) + "%)")

        except Exception as e:
            print("  [ERR] " + filename + ": " + str(e)[:70])

# ── Save CSV + JSON ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SAVING RESULTS...")
print("=" * 70)

csv_path  = os.path.join(OUTPUT_DIR, "test_results.csv")
json_path = os.path.join(OUTPUT_DIR, "test_results.json")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print("[OK] CSV  -> " + csv_path)

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("[OK] JSON -> " + json_path)

# ── Metrics ───────────────────────────────────────────────────────────────────
df      = pd.DataFrame(results)
total   = len(df)
correct = int(df["correct"].sum())
acc     = correct / total * 100 if total else 0

print("\nOverall accuracy: " + str(correct) + "/" + str(total) + " (" + str(round(acc, 1)) + "%)")

# by folder
for folder, grp in df.groupby("folder"):
    n   = len(grp)
    c   = int(grp["correct"].sum())
    pct = round(c / n * 100, 1) if n else 0
    print("  " + folder.ljust(20) + str(c) + "/" + str(n) + " (" + str(pct) + "%)")

# ── Visualizations ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CREATING VISUALIZATIONS...")
print("=" * 70)

# Tách riêng df 6 ngành chính (folder bắt đầu bằng "6_nganh_chinh/")
df6 = df[df["folder"].str.startswith("6_nganh_chinh/")].copy()
df6_correct = int(df6["correct"].sum())
df6_total   = len(df6)
df6_acc     = df6_correct / df6_total * 100 if df6_total else 0

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    "Danh gia model tren du lieu thuc te ngoai tap huan luyen (Out-of-Sample Evaluation)\n"
    "Pipeline: PDF/OCR -> BERT Encoder -> MLP Head -> Ket qua  |  60 CV x 6 nganh",
    fontsize=11, fontweight="bold"
)

# Plot 1: Accuracy từng ngành trong 6_nganh_chinh
ax = axes[0]
per_class = df6.groupby("ground_truth")["correct"].agg(["sum", "count"]).reset_index()
per_class["accuracy"] = (per_class["sum"] / per_class["count"] * 100).round(1)
per_class = per_class.sort_values("accuracy", ascending=True)
colors = ["#3b82f6" if x >= 80 else "#f59e0b" if x >= 60 else "#ef4444"
          for x in per_class["accuracy"]]
bars = ax.barh(per_class["ground_truth"], per_class["accuracy"], color=colors, height=0.6)
ax.set_xlabel("Accuracy (%)", fontsize=10)
ax.set_title("Accuracy tung nganh\n(10 CV/nganh, tong 60 CV)", fontsize=10, fontweight="bold")
ax.set_xlim(0, 115)
ax.axvline(x=df6_acc, color="black", linestyle="--", linewidth=1, alpha=0.6)
ax.text(df6_acc + 1, -0.5, "TB: " + str(round(df6_acc, 1)) + "%", fontsize=8, color="black")
for i, (v, row) in enumerate(zip(per_class["accuracy"], per_class.itertuples())):
    ax.text(v + 2, i, str(int(row.sum)) + "/" + str(row.count) + "  (" + str(v) + "%)",
            va="center", fontsize=9)

# Plot 2: Confusion matrix 6 ngành chính
ax = axes[1]
if len(df6) > 0:
    cm = confusion_matrix(df6["ground_truth"], df6["prediction"], labels=CLASSES)
    labels_short = ["BD", "CONS", "DM", "IT", "PR", "SALES"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_short, yticklabels=labels_short,
                ax=ax, cbar=True, linewidths=0.5)
    ax.set_title("Confusion Matrix -- 6 Nganh Chinh\n(60 CV thuc te)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Ground Truth", fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9)
    # Chú thích viết tắt
    legend_text = "BD=Business-Dev  CONS=Consultant\nDM=Digital-Media  IT=Info-Tech\nPR=Public-Relations"
    ax.text(0.02, -0.18, legend_text, transform=ax.transAxes, fontsize=7.5,
            verticalalignment="top", color="gray")
else:
    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

# Plot 3: Confidence phân phối đúng vs sai (chỉ df6)
ax = axes[2]
df6_ok  = df6[df6["correct"] == True]
df6_err = df6[df6["correct"] == False]
ax.hist(df6_ok["confidence"],  bins=10, alpha=0.75, color="#22c55e",
        label="Dung (" + str(len(df6_ok)) + "/" + str(df6_total) + ")")
ax.hist(df6_err["confidence"], bins=10, alpha=0.75, color="#ef4444",
        label="Sai  (" + str(len(df6_err)) + "/" + str(df6_total) + ")")
ax.axvline(x=50, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Confidence (%)", fontsize=10)
ax.set_ylabel("So luong CV", fontsize=10)
ax.set_title("Phan phoi Confidence\n(Nhan dung vs Nhan sai)", fontsize=10, fontweight="bold")
ax.legend(fontsize=9)
ax.text(0.5, 0.97,
        "Accuracy: " + str(df6_correct) + "/" + str(df6_total) + " = " + str(round(df6_acc, 1)) + "%",
        transform=ax.transAxes, ha="center", va="top", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fdf4", edgecolor="#22c55e"))

plt.tight_layout(rect=[0, 0.04, 1, 1])
report_path = os.path.join(OUTPUT_DIR, "test_evaluation_report.png")
plt.savefig(report_path, dpi=150, bbox_inches="tight")
print("[OK] Report -> " + report_path)
plt.close()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Pipeline: OCR/PDF -> BERT Encoder -> MLP Head -> Ket qua")
print("Model:    " + bert_config.get("encoder_name", "N/A"))
print("\n6 Nganh Chinh (60 CV thuc te):")
print("  Correct: " + str(df6_correct) + "/" + str(df6_total) + " (" + str(round(df6_acc, 1)) + "%)")
for _, row in per_class.sort_values("ground_truth").iterrows():
    print("  " + str(row["ground_truth"]).ljust(28) + str(int(row["sum"])) + "/" + str(row["count"]) +
          " (" + str(row["accuracy"]) + "%)")
print("\nOutput:")
print("  * " + csv_path)
print("  * " + json_path)
print("  * " + report_path)
print("=" * 70)
