import spectral.io.envi as envi
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import pandas as pd
import csv
import gc
import os

print("libraries Imported Successfully")

# ---------------------------------------------------------
# 0. MODEL ARCHITECTURE (EXACT MATCH TO CODE 1)
# ---------------------------------------------------------
class CNN1D(nn.Module):
    def __init__(self, input_len=1501, num_classes=9):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(47872, 256) 
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# -----------------------------
# 1. Setup & Metadata
# -----------------------------
hdr_path = r"C:\Users\20614\Downloads\20260224T070948492662\MROCR_4001_part_0003\mrocr_4001\mtrdr\2008\2008_060\frt0000a2c2\frt0000a2c2_07_if166j_mtr3.hdr"
cube = envi.open(hdr_path)
rows, cols, bands = cube.shape

wavelengths = np.array(cube.bands.centers, dtype=np.float32)
if wavelengths.max() > 100:
    wavelengths = wavelengths / 1000.0

# RANGE MATCHED TO CODE 1
wl_min, wl_max = 1.3, 2.6 

mineral_names = {0: "Carbonate", 1: "Inosil", 2: "Nesosil", 3: "Oxide", 4: "Phosphate", 5: "Phylosil", 6: "Sorosil", 7: "Sulfate", 8: "Tectosil", 9: "No Class"}
custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#000000']
discrete_cmap = ListedColormap(custom_colors)
discrete_cmap.set_bad('white')

# --- THE HARD FIX FOR THE ATTRIBUTE ERROR ---
model = CNN1D(input_len=1501, num_classes=9)

# We use weights_only=True to bypass the _utils check entirely
try:
    state_dict = torch.load("mineral_classifier_1501.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    print("Model Loaded")
except Exception:
    # If that fails, it's a version mismatch. We use the non-zip loading method.
    import io
    with open("mineral_classifier_1501.pth", 'rb') as f:
        buffer = io.BytesIO(f.read())
    state_dict = torch.load(buffer, map_location="cpu")
    model.load_state_dict(state_dict)
    print("Model Loaded")

model.eval()

confidence_threshold = 0.60
new_waves = np.linspace(wl_min, wl_max, 1501)
spectral_headers = [f"{w:.2f}um" for w in new_waves]
full_headers = ["row", "col"] + spectral_headers
csv_temp = "frt0000a2c2_07_if166j_mtr3_spectral_data_truncated.csv"

# -----------------------------
# 2. Part-by-Part Processing (Preserving Code 2 Structure)
# -----------------------------
num_parts = 4
rows_per_part = rows // num_parts
temp_files = []

with open(csv_temp, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(full_headers)

    for i in range(num_parts):
        start_r = i * rows_per_part
        end_r = rows if i == num_parts - 1 else (i + 1) * rows_per_part
        
        print(f"\n--- Processing Part {i+1}/{num_parts} (Rows {start_r} to {end_r}) ---")
        data_part = cube.read_subregion((start_r, end_r), (0, cols))
        part_rows = data_part.shape[0]
        classified_part = np.full((part_rows, cols), np.nan)

        with torch.no_grad():
            for r in range(part_rows):
                for c in range(cols):
                    spectrum = data_part[r, c, :].ravel().copy().astype(np.float32)
                    
                    if np.all(spectrum == 65535) or np.all(spectrum <= 0) or np.all(np.isnan(spectrum)):
                        continue 

                    spectrum[spectrum == 65535] = np.nan
                    
                    # Range filtering and normalization from Code 1
                    valid_mask = (~np.isnan(spectrum)) & \
                                 (spectrum <= 1.0) & (spectrum >= 0.0) & \
                                 (wavelengths >= wl_min) & (wavelengths <= wl_max)
                    
                    if np.sum(valid_mask) < 5: continue
                        
                    interp_func = interp1d(wavelengths[valid_mask], spectrum[valid_mask], 
                                           kind='linear', fill_value="extrapolate")
                    clean_spectrum = interp_func(new_waves)
                    
                    # CODE 1 MATH LOGIC
                    clean_spectrum = np.clip(clean_spectrum, 0, None)
                    clean_spectrum = clean_spectrum / (np.max(clean_spectrum) + 1e-12)

                    # Prediction
                    x = torch.tensor(clean_spectrum, dtype=torch.float32).view(1, 1, 1501)
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)
                    max_prob, pred_idx = torch.max(probs, dim=1)

                    if max_prob.item() >= confidence_threshold:
                        classified_part[r, c] = pred_idx.item()
                        writer.writerow([r + start_r, c] + list(clean_spectrum))
                    else:
                        classified_part[r, c] = 9

        temp_filename = f"frt0000a2c2_07_if166j_mtr3_spectral_data_truncated_part_{i}.npy"
        np.save(temp_filename, classified_part)
        temp_files.append(temp_filename)
        
        del data_part, classified_part
        gc.collect()

print("Processing Done. Data saved to CSV and .npy parts.")