# CONVERTING .SLI.HDR INTO  EXCEL

import spectral
import numpy as np
import pandas as pd

# Load spectral library
lib = spectral.open_image('phylosil.sli.hdr')

# Access data and metadata
data = np.array(lib.spectra, dtype=float)
wavelengths = np.array(lib.bands.centers)
names = lib.names

# -----------------------------
# Replace NaN and invalid values with 0
# -----------------------------

# Convert to numpy array
data = np.array(data, dtype=np.float32)

# Condition 1: Replace NaN with 0
data[np.isnan(data)] = 0.0

# Condition 2: Ensure zeros are 0 
data[data == 0] = 0.0
print("NaN and invalid values replaced with 0")


# Create DataFrame (rows = wavelengths, columns = minerals)
df = pd.DataFrame(data.T, index=wavelengths, columns=names)

# Save to Excel
output_path = "spectral_signatures_phylosil.xlsx"
df.to_excel(output_path)

print(f"Excel file saved as: {output_path}")
