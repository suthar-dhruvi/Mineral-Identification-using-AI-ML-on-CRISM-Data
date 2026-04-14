import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------- SETTINGS -----------
excel_file = "noisy_phylosil.xlsx"
start = 0     # starting spectrum index (0 = 1st spectrum)
end = 128       # ending spectrum (exclusive)

# ----------- LOAD DATA -----------
df = pd.read_excel(excel_file)

# Wavelength column = first column
wavelengths = df.iloc[:, 0].values

# All remaining columns = noisy spectra
spectra = df.iloc[:, 1:].values.T
# shape becomes: (num_spectra, num_wavelengths)

# Names of spectra (mineral names)
names = df.columns[1:]

num_spectra = spectra.shape[0]

# Safety check
if end > num_spectra:
    end = num_spectra

# ----------- PLOT SELECTED RANGE -----------
plt.figure(figsize=(12,6))

for i in range(start, end):
    plt.plot(wavelengths, spectra[i], label=names[i])

plt.xlabel("Wavelength (µm)")
plt.ylabel("Reflectance")
plt.title(f"Selected Spectra (Noisy) — {start+1} to {end}")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(True)
plt.show()
