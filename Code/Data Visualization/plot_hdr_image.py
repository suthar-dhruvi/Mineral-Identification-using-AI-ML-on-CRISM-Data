import spectral
import matplotlib.pyplot as plt

# Open spectral library (.sli + .hdr)
lib = spectral.open_image('phylosil.sli.hdr')

# Access the data and metadata directly
data = lib.spectra          # Reflectance data (shape = [89, 1501])
wavelengths = lib.bands.centers  # Wavelength values
names = lib.names           # List of mineral names


plt.figure(figsize=(12,6))

# Plot first 5 spectra
for i in range(128):
    plt.plot(wavelengths, data[i], label=names[i])


plt.xlabel('Wavelength (µm)')
plt.ylabel('Reflectance')
plt.title('Spectral Signatures from Library')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.show()