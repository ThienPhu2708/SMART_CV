"""
Train BERT-based CV classifier (SmartCV_BERT).
Pipeline:
  1. Load cleaned_resume.csv (chạy main.py trước nếu chưa có)
  2. Encode toàn bộ text bằng multilingual sentence-transformer (cache .npy)
  3. Split train / val (stratified 80/20)
  4. Train classification head với class-weighted CrossEntropyLoss
  5. Early stopping trên Val Macro F1
  6. Lưu model → Models/bert_classifier.pth + Models/bert_config.json

Chạy: python train_bert.py
"""
import os, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from SRC.model_bert import SmartCV_BERT
from SRC.Processing.bert_encoder import encode_texts, MODEL_NAME

# ── Cấu hình ──────────────────────────────────────────────────────────────────
CLEANED_CSV   = "Data/Processed/cleaned_resume.csv"
EMBED_CACHE   = "Data/Processed/bert_embeddings.npy"
LABEL_CACHE   = "Data/Processed/bert_labels.npy"
MODEL_OUT     = "Models/bert_classifier.pth"
CONFIG_OUT    = "Models/bert_config.json"
CURVES_OUT    = "Models/bert_training_curves.png"

EPOCHS        = 80
BATCH_SIZE    = 32
LR            = 2e-4
WEIGHT_DECAY  = 8e-3
LABEL_SMOOTH  = 0.1
PATIENCE      = 15
TEST_SIZE     = 0.15          # 15% test — đồng bộ với MLP để so sánh công bằng
VAL_RATIO     = 0.176         # 15% of total = 15/85 ≈ 17.6% of remaining after test split
RANDOM_STATE  = 42

# ── Step 1: Load data ─────────────────────────────────────────────────────────
print("=" * 60)
print("SmartCV BERT Training")
print("=" * 60)

if not os.path.exists(CLEANED_CSV):
    raise FileNotFoundError(
        f"Không tìm thấy {CLEANED_CSV}. Hãy chạy 'python main.py' trước!"
    )

df = pd.read_csv(CLEANED_CSV).dropna(subset=["cleaned_resume", "Category"])
df = df[df["cleaned_resume"].str.strip() != ""]
print(f"\nDataset: {len(df)} mẫu, {df['Category'].nunique()} ngành")
print(df["Category"].value_counts().to_string())

texts      = df["cleaned_resume"].tolist()
categories = df["Category"].tolist()

# ── Step 2: Encode (cache để tránh encode lại mỗi lần) ───────────────────────
le = LabelEncoder()
y  = le.fit_transform(categories)
classes = le.classes_

if os.path.exists(EMBED_CACHE) and os.path.exists(LABEL_CACHE):
    cached_y = np.load(LABEL_CACHE)
    if len(cached_y) == len(y) and np.array_equal(cached_y, y):
        print(f"\n[Cache] Đọc embeddings từ {EMBED_CACHE}")
        X = np.load(EMBED_CACHE)
    else:
        print("\n[Cache] Dataset thay đổi — encode lại...")
        X = encode_texts(texts)
        np.save(EMBED_CACHE, X)
        np.save(LABEL_CACHE, y)
else:
    print(f"\nEncoding {len(texts)} CV bằng '{MODEL_NAME}' ...")
    X = encode_texts(texts)
    np.save(EMBED_CACHE, X)
    np.save(LABEL_CACHE, y)
    print(f"Đã cache: {EMBED_CACHE}")

print(f"Embedding shape: {X.shape}")

# ── Step 3: Train / Val / Test split (70 / 15 / 15) — đồng bộ với MLP ───────
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=VAL_RATIO, random_state=RANDOM_STATE, stratify=y_temp
)
print(f"\nPhân chia dữ liệu (đồng bộ với MLP):")
print(f"  Train : {len(X_train):5d} mẫu ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Val   : {len(X_val):5d} mẫu ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Test  : {len(X_test):5d} mẫu ({len(X_test)/len(X)*100:.1f}%)")

# ── Step 4: DataLoader ────────────────────────────────────────────────────────
X_tr_t = torch.tensor(X_train, dtype=torch.float32)
y_tr_t = torch.tensor(y_train, dtype=torch.long)
X_va_t = torch.tensor(X_val,   dtype=torch.float32)
y_va_t = torch.tensor(y_val,   dtype=torch.long)
X_te_t = torch.tensor(X_test,  dtype=torch.float32)
y_te_t = torch.tensor(y_test,  dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_tr_t, y_tr_t),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=False
)

# ── Step 5: Class weights ─────────────────────────────────────────────────────
counts       = np.bincount(y_train, minlength=len(classes))
class_weights = torch.tensor(
    len(y_train) / (len(classes) * counts.clip(1)), dtype=torch.float32
)
print(f"\nClass weights: {dict(zip(classes, class_weights.numpy().round(3)))}")

# ── Step 6: Model + optimizer ─────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model     = SmartCV_BERT(embed_dim=X.shape[1], num_classes=len(classes)).to(device)
criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(device), label_smoothing=LABEL_SMOOTH
)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
# ReduceLROnPlateau: giảm LR khi Val F1 không cải thiện → ổn định hơn CosineAnnealing
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Classifier head: {total_params:,} params (encoder frozen)")

# ── Step 7: Training loop ─────────────────────────────────────────────────────
print(f"\nBắt đầu train {EPOCHS} epochs (patience={PATIENCE})...\n")

history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
best_val_f1  = 0.0
best_state   = None
epochs_no_imp = 0

X_va_dev = X_va_t.to(device)
y_va_dev = y_va_t.to(device)

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item() * len(yb)
        correct      += (out.argmax(1) == yb).sum().item()
        total        += len(yb)

    train_loss = running_loss / total
    train_acc  = correct / total

    # Validate
    model.eval()
    with torch.no_grad():
        val_out  = model(X_va_dev)
        val_loss = criterion(val_out, y_va_dev).item()
        val_pred = val_out.argmax(1).cpu().numpy()

    val_acc = accuracy_score(y_val, val_pred)
    val_f1  = f1_score(y_val, val_pred, average="macro", zero_division=0)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc * 100)
    history["val_acc"].append(val_acc * 100)
    history["val_f1"].append(val_f1 * 100)

    scheduler.step(val_f1)   # ReduceLROnPlateau: giảm LR khi val_f1 không cải thiện

    improved = ""
    if val_f1 > best_val_f1:
        best_val_f1  = val_f1
        best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        epochs_no_imp = 0
        improved = " ← BEST"
    else:
        epochs_no_imp += 1

    if epoch % 5 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Loss {train_loss:.4f}/{val_loss:.4f} | "
            f"Acc {train_acc*100:.1f}%/{val_acc*100:.1f}% | "
            f"F1 {val_f1*100:.2f}%{improved}"
        )

    if epochs_no_imp >= PATIENCE:
        print(f"\nEarly stopping tại epoch {epoch} (patience={PATIENCE})")
        break

# ── Step 8: Save ──────────────────────────────────────────────────────────────
os.makedirs("Models", exist_ok=True)
torch.save(best_state, MODEL_OUT)

print(f"\n✓ Model saved: {MODEL_OUT}")
print(f"  Best Val F1  = {best_val_f1*100:.2f}%")
print(f"  Best Val Acc = {max(history['val_acc']):.2f}%")

# ── Step 9: Đánh giá cuối trên TEST SET (unbiased — đồng bộ với MLP) ─────────
print("\n" + "=" * 60)
print("ĐÁNH GIÁ CUỐI CÙNG TRÊN TEST SET (unbiased)")
print("=" * 60)

model.load_state_dict(best_state)
model.eval()
device_cpu = torch.device("cpu")
model.to(device_cpu)

X_te_dev = X_te_t.to(device_cpu)
with torch.no_grad():
    te_out   = model(X_te_dev)
    te_probs = torch.nn.functional.softmax(te_out, dim=1)
    te_pred  = te_out.argmax(1).cpu().numpy()

test_acc = accuracy_score(y_test, te_pred)
test_f1  = f1_score(y_test, te_pred, average="macro", zero_division=0)
test_wf1 = f1_score(y_test, te_pred, average="weighted", zero_division=0)

print(f"Test Accuracy    : {test_acc*100:.2f}%")
print(f"Test Macro F1    : {test_f1*100:.2f}%")
print(f"Test Weighted F1 : {test_wf1*100:.2f}%")
print(f"\nClassification Report (Test Set):")
print(classification_report(y_test, te_pred, target_names=classes, zero_division=0))

config = {
    "encoder_name":   MODEL_NAME,
    "embed_dim":      int(X.shape[1]),
    "num_classes":    int(len(classes)),
    "classes":        list(classes),
    "best_val_f1":    round(best_val_f1 * 100, 4),
    "best_val_acc":   round(float(max(history["val_acc"])), 4),
    "test_accuracy":  round(test_acc * 100, 4),
    "test_macro_f1":  round(test_f1 * 100, 4),
    "split":          "70/15/15 (train/val/test)",
}
with open(CONFIG_OUT, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print(f"\n✓ Config saved: {CONFIG_OUT}")

# ── Step 10: Training curves ─────────────────────────────────────────────────
ep = range(1, len(history["train_loss"]) + 1)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(ep, history["train_loss"], label="Train")
axes[0].plot(ep, history["val_loss"],   label="Val")
axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(ep, history["train_acc"], label="Train")
axes[1].plot(ep, history["val_acc"],   label="Val")
axes[1].set_title("Accuracy (%)"); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(ep, history["val_f1"], color="purple", label="Val Macro F1")
axes[2].axhline(y=best_val_f1*100, color="red", linestyle="--",
                label=f"Best {best_val_f1*100:.2f}%")
axes[2].set_title("Val Macro F1 (%)"); axes[2].legend(); axes[2].grid(alpha=0.3)

plt.suptitle(f"BERT Classifier Training — {MODEL_NAME}", fontsize=11)
plt.tight_layout()
plt.savefig(CURVES_OUT, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✓ Training curves: {CURVES_OUT}")
print("\n=== BERT Training hoàn tất! ===")
print("Khởi động lại web app để dùng BERT model: python run_app.py")
