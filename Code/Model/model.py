# ======================================nitrates class kadhyo che 1501 and validated ======================================

# 1. IMPORTS
# ======================================
print("Importing libraries...")

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

print("Imports done\n")


# ======================================
# 2. FILE LIST (WITHOUT NITRATES)
# ======================================
train_files = [
    "train_dataset_carbonate.xlsx",
    "train_dataset_inosil.xlsx",
    "train_dataset_nesosil.xlsx",
    "train_dataset_oxide.xlsx",
    "train_dataset_phosphate.xlsx",
    "train_dataset_phylosil.xlsx",
    "train_dataset_sorosil.xlsx",
    "train_dataset_sulfate.xlsx",
    "train_dataset_tectosil.xlsx"
]

label_files = [
    "train_labels_carbonate.xlsx",
    "train_labels_inosil.xlsx",
    "train_labels_nesosil.xlsx",
    "train_labels_oxide.xlsx",
    "train_labels_phosphate.xlsx",
    "train_labels_phylosil.xlsx",
    "train_labels_sorosil.xlsx",
    "train_labels_sulfate.xlsx",
    "train_labels_tectosil.xlsx"
]

# ======================================
# 3. LOAD AND MERGE DATA
# ======================================
print("Loading all datasets...")

X_list, y_list = [], []

for data_f, label_f in zip(train_files, label_files):
    print("Reading:", data_f)

    # Clean data during load
    X_df = pd.read_excel(data_f).fillna(0) 
    y_df = pd.read_excel(label_f).fillna(0)

    X_list.append(X_df.values)
    y_list.append(y_df.values.ravel())

X = np.vstack(X_list)
y = np.concatenate(y_list)

print("Total samples:", X.shape[0])
print("Spectral length:", X.shape[1])
print("Classes before remap:", np.unique(y), "\n")

# ===== FIX LABEL INDEXING =====
y = y.astype(int) - 1

# ===== REMAP LABELS TO CONTINUOUS RANGE =====
unique_labels = np.unique(y)
label_map = {old: new for new, old in enumerate(unique_labels)}
y = np.array([label_map[label] for label in y])

print("Labels after remap:", np.unique(y), "\n")


# ======================================
# 4. TENSORS
# ======================================
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)


# ======================================
# 5. DATASET
# ======================================
class SpectralDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]

train_loader = DataLoader(SpectralDataset(X, y), batch_size=32, shuffle=True)
print("Batches:", len(train_loader), "\n")


# ======================================
# 6. CNN MODEL
# ======================================
class CNN1D(nn.Module):
    def __init__(self, input_len, num_classes):
        super().__init__()
        # Layer 1: Captures broad spectral features (64 filters)
        self.conv1 = nn.Conv1d(1, 64, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        
        # Layer 2: Detects specific absorption dips  (128 filters)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Layer 3: Identifies complex mineral fingerprints (256 filters)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)  
        self.bn3 = nn.BatchNorm1d(256)

        # Automatic Flattening
        with torch.no_grad():
            temp = torch.zeros(1, 1, input_len)
            temp = self.pool(torch.relu(self.bn1(self.conv1(temp))))
            temp = self.pool(torch.relu(self.bn2(self.conv2(temp))))
            temp = self.pool(torch.relu(self.bn3(self.conv3(temp))))
            self.flat = temp.numel()

        self.fc1 = nn.Linear(self.flat, 256)
        self.dropout = nn.Dropout(0.4)       # 40% chance to drop a neuron======handles the overfitting problem. By randomly ignoring 40% of the neurons during training
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))     # Batch Normalization :-handle the stability problem. This ensures that even if one CRISM pixel is brighter or noisier than another, the model "normalizes" them to the same scale before trying to classify them.
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))     # Batch Normalization (self.bn1 ,self.bn2, self.bn3)
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))     # Batch Normalization
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

model = CNN1D(X.shape[1], len(np.unique(y)))
print("Model built\n")

# Initialize lists
train_losses, train_accs = [], []
val_losses, val_accs = [], []

# ======================================
# 7. TRAINING
# ======================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 40

print("Training started...\n")

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0

    # --- BATCH LOOP ---
    for i, (xb, yb) in enumerate(train_loader):
        optimizer.zero_grad()
        mask = (xb != 0).float()
        xb_masked = xb * mask
        outputs = model(xb_masked)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += yb.size(0)
        correct += (predicted == yb).sum().item()

    # --- VALIDATION STEP ---
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in train_loader: 
            outputs = model(xb)
            v_loss = criterion(outputs, yb)
            val_loss += v_loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += yb.size(0)
            val_correct += (predicted == yb).sum().item()

    # --- APPEND HISTORY ---
    train_losses.append(running_loss / len(train_loader))
    train_accs.append(correct / total)
    val_losses.append(val_loss / len(train_loader))
    val_accs.append(val_correct / val_total)

    # --- PRINT ONCE PER EPOCH ---
    print(f"Epoch {epoch+1}: "
          f"Train Loss={train_losses[-1]:.5f}, Train Acc={train_accs[-1]:.3f}, "
          f"Val Loss={val_losses[-1]:.5f}, Val Acc={val_accs[-1]:.3f}\n")

# --- SAVE ---
torch.save(model.state_dict(), "mineral_classifier_1501.pth")
print("Training finished. Model saved.")