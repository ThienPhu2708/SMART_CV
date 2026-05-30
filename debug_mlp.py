import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from SRC.model_mlp import SmartCV_MLP

# Load data
vectors_path = "Data/Processed/resume_vectors.csv"
labels_path = "Data/Processed/cleaned_resume.csv"

X = pd.read_csv(vectors_path).values.astype(np.float32)
df = pd.read_csv(labels_path)

le = LabelEncoder()
y = le.fit_transform(df["Category"])
classes = le.classes_

# Split data
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)

# Load model
import json
ckpt_log_path = "Models/checkpoints/log.json"
with open(ckpt_log_path, 'r') as f:
    ckpt_log = json.load(f)

best_ckpt = ckpt_log[0]
best_ckpt_path = os.path.join("Models/checkpoints", best_ckpt['file'])

input_dim = X.shape[1]
model = SmartCV_MLP(input_size=input_dim, num_classes=len(classes))
model.load_state_dict(torch.load(best_ckpt_path, map_location="cpu"))
model.eval()

# Inference
predictions = []
with torch.no_grad():
    for i in range(0, len(X_test), 64):
        batch = X_test[i:i+64]
        batch_tensor = torch.tensor(batch).float()
        outputs = model(batch_tensor)
        preds = torch.argmax(outputs, dim=1)
        predictions.extend(preds.numpy())

predictions = np.array(predictions)

# Check report structure
report = classification_report(y_test, predictions, target_names=classes, digits=4, output_dict=True)

print("Report keys:", list(report.keys()))
print("\nReport structure:")
for key in report.keys():
    print(f"  {key}: {report[key]}")

print(f"\nPrediction unique classes: {np.unique(predictions)}")
print(f"True test unique classes: {np.unique(y_test)}")
print(f"Number of test samples: {len(y_test)}")
print(f"Predicted class distribution:")
for i, cls in enumerate(classes):
    count = np.sum(predictions == i)
    print(f"  {cls}: {count}")
