# ======================================
# Prediction script for mineral classifier (without Nitrates====Overall accuracy=======mineral_classifier_1501)
# ======================================
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print("Imports complete\n")

# ======================================
# 2. TEST FILE LIST
# ======================================
test_files = [
    "test_dataset_carbonate.xlsx", "test_dataset_inosil.xlsx", "test_dataset_nesosil.xlsx",
    "test_dataset_oxide.xlsx", "test_dataset_phosphate.xlsx", "test_dataset_phylosil.xlsx",
    "test_dataset_sorosil.xlsx", "test_dataset_sulfate.xlsx", "test_dataset_tectosil.xlsx"
]

test_label_files = [
    "test_labels_carbonate.xlsx", "test_labels_inosil.xlsx", "test_labels_nesosil.xlsx",
    "test_labels_oxide.xlsx", "test_labels_phosphate.xlsx", "test_labels_phylosil.xlsx",
    "test_labels_sorosil.xlsx", "test_labels_sulfate.xlsx", "test_labels_tectosil.xlsx"
]

# ======================================
# 3. LOAD TEST DATA
# ======================================
print("Loading test data...")

X_list, y_list = [], []

for data_f, label_f in zip(test_files, test_label_files):
    if not os.path.exists(data_f) or not os.path.exists(label_f):
        print(f"Skipping {data_f} - file not found.")
        continue

    X_df = pd.read_excel(data_f).fillna(0)
    y_df = pd.read_excel(label_f).fillna(0)

    # CRITICAL FIX: Ensure row matching for every file pair to prevent AssertionError
    rows = min(len(X_df), len(y_df))
    X_list.append(X_df.values[:rows])
    y_list.append(y_df.values.ravel()[:rows])

X = np.vstack(X_list)
y = np.concatenate(y_list)

print("Samples:", X.shape[0])
print("Bands:", X.shape[1])

# ===== LABEL REMAP (Matches Training Logic) =====
unique_labels = np.unique(y)
label_map = {old: new for new, old in enumerate(unique_labels)}
y = np.array([label_map[label] for label in y])

print("Labels after remap:", np.unique(y), "\n")

# Clean spectra
X = np.nan_to_num(X, nan=0.0)
X = np.clip(X, 0, None)

# Convert to tensors
X_test = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # shape: (N, 1, bands)
y_test = torch.tensor(y, dtype=torch.long)

# Final check for TensorDataset alignment
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
print("Test loader ready. Batches:", len(test_loader), "\n")

# ======================================
# 4. MODEL (Matches Code 2 Architecture)
# ======================================
class CNN1D(nn.Module):
    def __init__(self, input_len, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)  
        self.bn3 = nn.BatchNorm1d(256)

        with torch.no_grad():
            temp = torch.zeros(1, 1, input_len)
            temp = self.pool(torch.relu(self.bn1(self.conv1(temp))))
            temp = self.pool(torch.relu(self.bn2(self.conv2(temp))))
            temp = self.pool(torch.relu(self.bn3(self.conv3(temp))))
            self.flat = temp.numel()

        self.fc1 = nn.Linear(self.flat, 256)
        self.dropout = nn.Dropout(0.5) # Updated to 0.5 to match final training
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

num_classes = len(np.unique(y))
model = CNN1D(input_len=X_test.shape[-1], num_classes=num_classes)

# Load saved weights
model_path = "mineral_classifier_1501_balanced.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print("Model loaded successfully.\n")
else:
    print(f"Warning: {model_path} not found. Check filename.")

# ======================================
# 5. EVALUATE MODEL
# ======================================
print("Evaluating model on test set...")

y_pred, y_true = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        # Apply mask as done in training
        mask = (inputs != 0).float()
        outputs = model(inputs * mask)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# 1. Total Accuracy (Syntax fixed)
total_correct = np.sum(y_true == y_pred)
total_samples = len(y_true)
overall_accuracy = (total_correct / total_samples) * 100

print("-" * 30)
print(f"OVERALL MODEL ACCURACY: {overall_accuracy:.2f}%")
print("-" * 30)

# 2. Class-wise Accuracy
cm = confusion_matrix(y_true, y_pred)
class_names = ["Carbonate", "Inosil", "Nesosil", "Oxide", "Phosphate", "Phylosil", "Sorosil", "Sulfate", "Tectosil"]

print("\nCLASS-WISE ACCURACY:")
# Added handling for potential missing classes in test set
row_sums = cm.sum(axis=1)
for i, name in enumerate(class_names):
    if i < len(row_sums) and row_sums[i] > 0:
        acc = cm[i, i] / row_sums[i]
        print(f"{name.ljust(12)} : {acc*100:6.2f}%")

# 3. Visualization
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names[:len(cm)])
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, xticks_rotation=45)
plt.title(f"Confusion Matrix (Overall Acc: {overall_accuracy:.2f}%)")
plt.tight_layout()
plt.show()