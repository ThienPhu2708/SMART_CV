"""
Script để evaluate MLP model trên test set
Lấy metrics và so sánh với BERT
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from SRC.model_mlp import SmartCV_MLP

# Paths
DATA_PATH = "Data/Processed"
MODEL_PATH = "Models/smartcv_model.pth"
CLASSES_PATH = "Models/classes.npy"
VECTORIZER_PATH = "Data/Processed/tfidf_vectorizer.pkl"

print("=" * 70)
print("MLP MODEL EVALUATION ON TEST SET")
print("=" * 70)

# Load data
print("\n[1] Loading test data...")
X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
y_test = np.load(os.path.join(DATA_PATH, "y_test.npy"))
classes = np.load(CLASSES_PATH, allow_pickle=True)

print(f"    Test set size: {len(X_test)} samples")
print(f"    Classes: {classes}")
print(f"    Class distribution:\n{pd.Series(y_test).value_counts().sort_index()}")

# Load vectorizer to get input dim
print("\n[2] Loading TF-IDF vectorizer...")
with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)
input_dim = len(vectorizer.get_feature_names_out())
print(f"    Input dimension: {input_dim}")

# Load model
print("\n[3] Loading MLP model...")
model = SmartCV_MLP(input_dim=input_dim, num_classes=len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print(f"    Model loaded from {MODEL_PATH}")

# Run inference
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

        if (i // 64 + 1) % 5 == 0:
            print(f"    Processed {min(i+64, len(X_test))}/{len(X_test)} samples")

predictions = np.array(predictions)
probabilities = np.array(probabilities)

# Calculate metrics
print("\n[5] Calculating metrics...")
test_acc = accuracy_score(y_test, predictions) * 100
test_f1_macro = f1_score(y_test, predictions, average='macro') * 100
test_f1_weighted = f1_score(y_test, predictions, average='weighted') * 100

print(f"\n{'='*70}")
print("TEST SET RESULTS FOR MLP")
print(f"{'='*70}")
print(f"Test Accuracy:      {test_acc:.4f}%")
print(f"Test Macro F1:      {test_f1_macro:.4f}%")
print(f"Test Weighted F1:   {test_f1_weighted:.4f}%")

# Per-class metrics
print(f"\n{'='*70}")
print("PER-CLASS METRICS (Test Set)")
print(f"{'='*70}")
report = classification_report(y_test, predictions, target_names=classes, digits=4, output_dict=True)

class_metrics = []
for i, class_name in enumerate(classes):
    precision = report[str(i)]['precision'] * 100
    recall = report[str(i)]['recall'] * 100
    f1 = report[str(i)]['f1-score'] * 100
    support = int(report[str(i)]['support'])

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
    'Metric': ['Test Accuracy', 'Test Macro F1', 'Test Weighted F1'],
    'MLP': [f"{test_acc:.2f}%", f"{test_f1_macro:.2f}%", f"{test_f1_weighted:.2f}%"],
    'BERT': [
        f"{bert_config['test_accuracy']:.2f}%",
        f"{bert_config['test_macro_f1']:.2f}%",
        'N/A'
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Save results
print(f"\n{'='*70}")
print("SAVING RESULTS")
print(f"{'='*70}")

results = {
    'model': 'MLP (TF-IDF)',
    'test_accuracy': float(test_acc),
    'test_macro_f1': float(test_f1_macro),
    'test_weighted_f1': float(test_f1_weighted),
    'per_class_metrics': class_metrics,
    'test_set_size': len(X_test),
    'num_classes': len(classes),
    'classes': classes.tolist()
}

with open('Models/mlp_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to: Models/mlp_test_results.json")

# Print summary for thesis
print(f"\n{'='*70}")
print("COPY THIS TO THESIS:")
print(f"{'='*70}")
print(f"""
MLP Test Results:
- Test Accuracy: {test_acc:.2f}%
- Test Macro F1: {test_f1_macro:.2f}%

BERT Test Results (from bert_config.json):
- Test Accuracy: {bert_config['test_accuracy']:.2f}%
- Test Macro F1: {bert_config['test_macro_f1']:.2f}%

Best MLP Checkpoint (from checkpoints/log.json):
- Epoch: 13
- Val F1: 88.25%
- Val Accuracy: 90.53%
""")

print(f"\n✓ Evaluation complete!")
