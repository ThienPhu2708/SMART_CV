"""
Script để evaluate MLP model trên test set
Replicate train/val/test split từ train.py
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from SRC.model_mlp import SmartCV_MLP

print("=" * 70)
print("MLP MODEL EVALUATION ON TEST SET")
print("=" * 70)

# Load data (giống như train.py)
print("\n[1] Loading data...")
vectors_path = "Data/Processed/resume_vectors.csv"
labels_path = "Data/Processed/cleaned_resume.csv"

X = pd.read_csv(vectors_path).values.astype(np.float32)
df = pd.read_csv(labels_path)

print(f"    Loaded: {len(X)} samples × {X.shape[1]} features")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df["Category"])
classes = le.classes_

print(f"    Classes: {classes}")

# Split data (70/15/15) - replicate train.py exactly
print("\n[2] Splitting data (70/15/15 stratified)...")
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)

print(f"    Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"    Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"    Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# Load best MLP model from checkpoint
print("\n[3] Loading best MLP checkpoint...")
ckpt_log_path = "Models/checkpoints/log.json"
with open(ckpt_log_path, 'r') as f:
    ckpt_log = json.load(f)

# Get best checkpoint (first one is best)
best_ckpt = ckpt_log[0]
best_ckpt_path = os.path.join("Models/checkpoints", best_ckpt['file'])
print(f"    Best checkpoint: {best_ckpt['file']}")
print(f"    Epoch: {best_ckpt['epoch']}, Val F1: {best_ckpt['val_f1']}%, Val Acc: {best_ckpt['val_acc']}%")

# Load model
input_dim = X.shape[1]
model = SmartCV_MLP(input_size=input_dim, num_classes=len(classes))
model.load_state_dict(torch.load(best_ckpt_path, map_location="cpu"))
model.eval()

# Run inference on test set
print("\n[4] Running inference on test set...")
predictions = []
probabilities = []

with torch.no_grad():
    for i in range(0, len(X_test), 64):
        batch = X_test[i:i+64]
        batch_tensor = torch.tensor(batch).float()
        outputs = model(batch_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        predictions.extend(preds.numpy())
        probabilities.extend(probs.numpy())

predictions = np.array(predictions)
probabilities = np.array(probabilities)

# Calculate metrics
print("\n[5] Calculating metrics...")
test_acc = accuracy_score(y_test, predictions) * 100
test_f1_macro = f1_score(y_test, predictions, average='macro') * 100
test_f1_weighted = f1_score(y_test, predictions, average='weighted') * 100
test_prec_macro = np.mean([
    accuracy_score(y_test[y_test == i], predictions[y_test == i])
    if sum(y_test == i) > 0 else 0
    for i in range(len(classes))
]) * 100

print(f"\n{'='*70}")
print("TEST SET RESULTS FOR MLP")
print(f"{'='*70}")
print(f"Test Accuracy:      {test_acc:.2f}%")
print(f"Test Macro F1:      {test_f1_macro:.2f}%")
print(f"Test Weighted F1:   {test_f1_weighted:.2f}%")

# Per-class metrics
print(f"\n{'='*70}")
print("PER-CLASS METRICS (Test Set)")
print(f"{'='*70}")
report = classification_report(y_test, predictions, target_names=classes, digits=4, output_dict=True)

class_metrics = []
for class_name in classes:
    if class_name in report:
        precision = report[class_name]['precision'] * 100
        recall = report[class_name]['recall'] * 100
        f1 = report[class_name]['f1-score'] * 100
        support = int(report[class_name]['support'])

        class_metrics.append({
            'Class': class_name,
            'Precision': f"{precision:.2f}%",
            'Recall': f"{recall:.2f}%",
            'F1-Score': f"{f1:.2f}%",
            'Support': support
        })

        print(f"{class_name:30s} | P: {precision:6.2f}% | R: {recall:6.2f}% | F1: {f1:6.2f}% | n={support}")

# Load BERT results for comparison
print(f"\n{'='*70}")
print("COMPARISON: MLP vs BERT (Test Set)")
print(f"{'='*70}")

with open("Models/bert_config.json", 'r') as f:
    bert_config = json.load(f)

comparison_data = {
    'Metric': ['Test Accuracy', 'Test Macro F1'],
    'MLP': [f"{test_acc:.2f}%", f"{test_f1_macro:.2f}%"],
    'BERT': [f"{bert_config['test_accuracy']:.2f}%", f"{bert_config['test_macro_f1']:.2f}%"]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Save results
print(f"\n{'='*70}")
print("SAVING RESULTS")
print(f"{'='*70}")

results = {
    'model': 'MLP (TF-IDF)',
    'checkpoint_epoch': best_ckpt['epoch'],
    'checkpoint_val_f1': best_ckpt['val_f1'],
    'checkpoint_val_acc': best_ckpt['val_acc'],
    'test_accuracy': float(test_acc),
    'test_macro_f1': float(test_f1_macro),
    'test_weighted_f1': float(test_f1_weighted),
    'per_class_metrics': class_metrics,
    'test_set_size': len(X_test),
    'num_classes': len(classes),
    'classes': classes.tolist()
}

with open('Models/mlp_test_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"[OK] Results saved to: Models/mlp_test_results.json")

# Print summary for thesis
print(f"\n{'='*70}")
print("[OK] SUMMARY FOR THESIS:")
print(f"{'='*70}")
print(f"""
MLP Validation Results (Best Checkpoint - Epoch {best_ckpt['epoch']}):
  - Val F1: {best_ckpt['val_f1']}%
  - Val Acc: {best_ckpt['val_acc']}%

MLP Test Results:
  - Test Accuracy: {test_acc:.2f}%
  - Test Macro F1: {test_f1_macro:.2f}%

BERT Test Results:
  - Test Accuracy: {bert_config['test_accuracy']:.2f}%
  - Test Macro F1: {bert_config['test_macro_f1']:.2f}%

Overfitting Analysis:
  - MLP: Train ~99% vs Test {test_acc:.2f}% (gap: ~{99-test_acc:.1f}%)
  - BERT: Train ~91% vs Test {bert_config['test_accuracy']:.2f}% (gap: ~{91-bert_config['test_accuracy']:.1f}%)
""")

print(f"[OK] Evaluation complete!")
