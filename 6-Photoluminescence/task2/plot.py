
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the .csv file
data_file = 'CNT.csv'  # Ensure the correct path
calib_file = 'calib.csv'  # Calibration file

# Read the main data file
data = np.genfromtxt(data_file, delimiter=';', skip_header=1)

# Extract x (Emission) and y (Excitation) labels
wavelengths_x = np.genfromtxt(data_file, delimiter=';', max_rows=1)[1:]
wavelengths_y = np.genfromtxt(data_file, delimiter=';', skip_header=1, usecols=0)

# Transpose the data to match correct axes
data = data.T
wavelengths_x, wavelengths_y = wavelengths_y, wavelengths_x

# Round emission wavelengths to whole numbers
wavelengths_x = np.round(wavelengths_x).astype(int)

# Load calibration data
calib_data = np.genfromtxt(calib_file, delimiter=';', skip_header=3)  # Skip header lines
calib_wavelengths = calib_data[:, 0]  # Wavelengths in calibration file
calib_counts = calib_data[:, 1]  # Corresponding counts

# Interpolate calibration values to match the emission wavelengths
calib_interp = np.interp(wavelengths_x, calib_wavelengths, calib_counts)

# Avoid division by zero
calib_interp[calib_interp == 0] = 1

# Normalize data
data /= calib_interp[np.newaxis, :]  # Divide each column by its corresponding calibration factor

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data, xticklabels=wavelengths_x, yticklabels=wavelengths_y, cmap='viridis', cbar=True)

# Adjust tick labels
plt.xticks(ticks=np.arange(0, len(wavelengths_x), step=10), labels=wavelengths_x[::10], rotation=90)
plt.yticks(ticks=np.arange(0, len(wavelengths_y), step=10), labels=wavelengths_y[::10], rotation=0)

# Invert y-axis
plt.gca().invert_yaxis()

# Labels and title
plt.xlabel('Emission Wavelength (nm)')
plt.ylabel('Excitation Wavelength (nm)')
plt.title('Calibrated 2D Heatmap of Count Intensity')

# Show plot
plt.tight_layout()
plt.show()
