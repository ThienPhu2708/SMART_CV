"""
SmartCV — Training Pipeline v2
================================
Cải thiện so với v1:
  - Class-weighted CrossEntropyLoss + Label Smoothing  → chống lệch lớp thiểu số
  - SMOTE thay RandomOverSampler                       → synthetic samples chất lượng hơn
  - Gradient Clipping                                  → ổn định quá trình huấn luyện
  - Track Macro F1 + Weighted F1                       → metric phù hợp cho imbalanced data
  - Lưu model tốt nhất theo Val Macro F1 (không phải val_loss)
  - Confusion Matrix visualization
  - Cosine Annealing LR với warm restart

Quy trình đúng chuẩn:
  1. Chia Train / Val / Test TRƯỚC (70 / 15 / 15) — stratified
  2. SMOTE CHỈ trên tập Train (tránh data leakage)
  3. Huấn luyện mini-batch qua DataLoader
  4. Chọn model tốt nhất dựa trên Val Macro F1
  5. Early Stopping để tránh overfitting
  6. Đánh giá cuối cùng 1 lần duy nhất trên Test Set (unbiased)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

from SRC.model_mlp import SmartCV_MLP

# ── Siêu tham số ──────────────────────────────────────────────────────────────
EPOCHS          = 150
BATCH_SIZE      = 64
LEARNING_RATE   = 0.001
PATIENCE        = 12       # early stopping: dừng nếu Val F1 không tăng sau N epoch
LABEL_SMOOTHING = 0.1      # giảm overconfidence của model
GRAD_CLIP       = 1.0      # gradient clipping — ổn định training
WEIGHT_DECAY    = 1e-3     # tăng L2 regularization để chống overfit
SMOTE_MAX       = 180      # giới hạn mẫu SMOTE mỗi class (không oversample quá mức)


# ── Dataset helper ─────────────────────────────────────────────────────────────
def make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(torch.tensor(X).float(), torch.tensor(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ── Vẽ Learning Curves ────────────────────────────────────────────────────────
def plot_curves(train_losses, val_losses, train_accs, val_accs,
                val_f1s, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(train_losses, label="Train Loss", color="#1f77b4")
    axes[0].plot(val_losses,   label="Val Loss",   color="#ff7f0e")
    axes[0].set_title("Loss theo Epoch")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot([a * 100 for a in train_accs], label="Train Acc", color="#1f77b4")
    axes[1].plot([a * 100 for a in val_accs],   label="Val Acc",   color="#ff7f0e")
    axes[1].set_title("Accuracy theo Epoch (%)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot([f * 100 for f in val_f1s], label="Val Macro F1", color="#2ca02c")
    axes[2].set_title("Val Macro F1 theo Epoch (%)")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Macro F1 (%)")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Đã lưu biểu đồ: {save_path}")


# ── Vẽ Confusion Matrix ───────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    n = len(class_names)
    fig_size = max(12, n * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (normalized)")

    thresh = cm_norm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i,j]}",
                    ha="center", va="center", fontsize=6,
                    color="white" if cm_norm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Đã lưu confusion matrix: {save_path}")


# ── Đánh giá một epoch ────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out  = model(X_batch)
            loss = criterion(out, y_batch)
            total_loss += loss.item() * len(y_batch)
            _, preds = torch.max(out, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1, all_preds, all_labels


# ── Pipeline chính ─────────────────────────────────────────────────────────────
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}")

    # ── 1. Nạp dữ liệu ──────────────────────────────────────────────────────
    vectors_path = "Data/Processed/resume_vectors.csv"
    labels_path  = "Data/Processed/cleaned_resume.csv"
    if not os.path.exists(vectors_path) or not os.path.exists(labels_path):
        print("Lỗi: Chưa có dữ liệu. Hãy chạy main.py trước!")
        return

    X  = pd.read_csv(vectors_path).values.astype(np.float32)
    df = pd.read_csv(labels_path)

    if len(X) != len(df):
        print("Lỗi: resume_vectors.csv và cleaned_resume.csv không khớp số dòng!")
        return

    input_dim = X.shape[1]
    print(f"Dữ liệu: {len(X)} mẫu | {input_dim} đặc trưng TF-IDF")

    # ── 2. Mã hóa nhãn ──────────────────────────────────────────────────────
    le = LabelEncoder()
    y  = le.fit_transform(df["Category"])
    num_classes = len(le.classes_)
    print(f"Số ngành nghề: {num_classes}")

    # Kiểm tra phân bố lớp
    unique, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    print(f"Lớp ít nhất: {le.classes_[counts.argmin()]} ({min_count} mẫu)")
    print(f"Lớp nhiều nhất: {le.classes_[counts.argmax()]} ({counts.max()} mẫu)")

    # ── 3. Chia tập TRƯỚC khi oversample — 70 / 15 / 15 ─────────────────────
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )
    print(f"\nPhân chia dữ liệu:")
    print(f"  Train : {len(X_train):5d} mẫu ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val   : {len(X_val):5d} mẫu ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test  : {len(X_test):5d} mẫu ({len(X_test)/len(X)*100:.1f}%)")

    # ── 4. SMOTE CHỈ trên Train — tránh data leakage ─────────────────────────
    # Giới hạn mỗi class tối đa SMOTE_MAX mẫu để không oversample quá mức
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    min_train_count = train_counts.min()
    k_neighbors = min(5, min_train_count - 1)

    # sampling_strategy: chỉ upsample các class có < SMOTE_MAX mẫu, lên đúng SMOTE_MAX
    smote_target = {
        int(cls): SMOTE_MAX
        for cls, count in zip(train_unique, train_counts)
        if count < SMOTE_MAX
    }

    if k_neighbors < 1:
        print(f"\nCảnh báo: Lớp thiểu số quá ít mẫu ({min_train_count}). Bỏ qua SMOTE.")
        X_train_res, y_train_res = X_train, y_train
    elif not smote_target:
        print(f"\nTất cả class đã >= {SMOTE_MAX} mẫu. Bỏ qua SMOTE.")
        X_train_res, y_train_res = X_train, y_train
    else:
        print(f"\nSMOTE: k_neighbors={k_neighbors}, target mỗi class = {SMOTE_MAX} mẫu tối đa")
        print(f"Trước SMOTE: {len(X_train)} mẫu train")
        try:
            smote = SMOTE(
                k_neighbors=k_neighbors,
                sampling_strategy=smote_target,
                random_state=42
            )
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            print(f"Sau  SMOTE : {len(X_train_res)} mẫu train (Val/Test giữ nguyên)")
        except Exception as e:
            print(f"SMOTE thất bại ({e}), dùng dữ liệu gốc.")
            X_train_res, y_train_res = X_train, y_train

    # ── 5. Tính Class Weights cho Loss ───────────────────────────────────────
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_train  # dùng y_train gốc (trước SMOTE) để tính weight thực tế
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"\nClass weights (min={class_weights.min():.2f}, max={class_weights.max():.2f})")

    # ── 6. Tạo DataLoader ────────────────────────────────────────────────────
    train_loader = make_loader(X_train_res, y_train_res, BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(X_val,       y_val,       BATCH_SIZE, shuffle=False)
    test_loader  = make_loader(X_test,      y_test,      BATCH_SIZE, shuffle=False)

    # ── 7. Khởi tạo model, loss, optimizer, scheduler ────────────────────────
    model = SmartCV_MLP(input_size=input_dim, num_classes=num_classes).to(device)

    # Label Smoothing + Class Weights: giảm overconfidence và cân bằng gradient
    criterion = nn.CrossEntropyLoss(
        weight=weights_tensor,
        label_smoothing=LABEL_SMOOTHING
    )

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Cosine Annealing: LR giảm mượt theo chu kỳ cosine thay vì giật cấp
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE * 0.01
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nKiến trúc MLP: {total_params:,} tham số")
    print(f"Loss: CrossEntropy + label_smoothing={LABEL_SMOOTHING} + class_weights")
    print(f"Optimizer: Adam (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    print(f"Scheduler: CosineAnnealingLR (T_max={EPOCHS})")

    # ── 8. Vòng lặp huấn luyện ───────────────────────────────────────────────
    best_val_f1    = -1.0
    best_val_loss  = float('inf')
    epochs_no_imp  = 0
    train_losses, val_losses  = [], []
    train_accs,   val_accs    = [], []
    val_f1s                   = []

    os.makedirs("Models", exist_ok=True)
    print(f"\nBắt đầu huấn luyện (tối đa {EPOCHS} epochs, "
          f"early stopping patience={PATIENCE})...\n")

    for epoch in range(1, EPOCHS + 1):
        # --- Train một epoch ---
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out  = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            # Gradient clipping — ngăn gradient bùng nổ
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item() * len(y_batch)

        train_loss = epoch_loss / len(train_loader.dataset)
        _, train_acc, train_f1, _, _ = evaluate(model, train_loader, criterion, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        # Log mỗi 10 epoch
        if epoch % 10 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch:3d}/{EPOCHS}] "
                  f"| Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.1f}%  F1: {train_f1*100:.1f}% "
                  f"| Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.1f}%  F1: {val_f1*100:.1f}% "
                  f"| LR: {current_lr:.6f}")

        # Lưu model tốt nhất theo Val Macro F1 (phù hợp imbalanced data)
        if val_f1 > best_val_f1:
            best_val_f1   = val_f1
            best_val_loss = val_loss
            epochs_no_imp = 0
            torch.save(model.state_dict(), "Models/smartcv_model.pth")
        else:
            epochs_no_imp += 1

        # Early Stopping
        if epochs_no_imp >= PATIENCE:
            print(f"\nEarly stopping tại epoch {epoch} "
                  f"(Val F1 không cải thiện {PATIENCE} epoch liên tiếp)")
            break

    # ── 9. Lưu artifacts ─────────────────────────────────────────────────────
    np.save("Models/classes.npy", le.classes_)
    plot_curves(train_losses, val_losses, train_accs, val_accs, val_f1s,
                "Models/training_curves.png")

    # ── 10. Đánh giá CUỐI CÙNG trên Test Set ────────────────────────────────
    print("\n" + "=" * 65)
    print("ĐÁNH GIÁ CUỐI CÙNG TRÊN TEST SET (unbiased)")
    print("=" * 65)

    model.load_state_dict(torch.load("Models/smartcv_model.pth", map_location=device))
    test_loss, test_acc, test_macro_f1, y_pred, y_true = evaluate(
        model, test_loader, criterion, device
    )
    test_weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"Test Loss            : {test_loss:.4f}")
    print(f"Test Accuracy        : {test_acc*100:.2f}%")
    print(f"Test Macro F1        : {test_macro_f1*100:.2f}%")
    print(f"Test Weighted F1     : {test_weighted_f1*100:.2f}%")
    print(f"\nBest Val Macro F1    : {best_val_f1*100:.2f}%")
    print(f"Best Val Loss        : {best_val_loss:.4f}")

    overfit_gap = (max(train_accs) - test_acc) * 100
    print(f"Overfit gap          : {overfit_gap:.2f}%  "
          f"({'OK' if overfit_gap < 10 else 'Cảnh báo: có thể overfit'})")

    print(f"\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=le.classes_, digits=3))

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, le.classes_,
                          "Models/confusion_matrix.png")

    print(f"\nModel đã lưu       : Models/smartcv_model.pth")
    print(f"Biểu đồ training   : Models/training_curves.png")
    print(f"Confusion matrix   : Models/confusion_matrix.png")


if __name__ == "__main__":
    train_model()
