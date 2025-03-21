
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d

# File paths
dlf_file = 'dlf.csv'
water_background_file = 'waterbackground.csv'
calib_file = 'calib.csv'

# Load the data
dlf_data = np.genfromtxt(dlf_file, delimiter=';', skip_header=1)
water_background_data = np.genfromtxt(water_background_file, delimiter=';', skip_header=1)

# Extract the x and y wavelengths
wavelengths_x_full = np.genfromtxt(dlf_file, delimiter=';', max_rows=1)[1:]  # Full Emission Wavelengths
wavelengths_y = np.genfromtxt(dlf_file, delimiter=';', skip_header=1, usecols=0)  # Excitation Wavelengths

wavelengths_x_bg = np.genfromtxt(water_background_file, delimiter=';', max_rows=1)[1:]  # Background Emission Wavelengths

# Convert to matrices
dlf_matrix = dlf_data[:, 1:]  # Exclude first column (excitation wavelengths)
water_background_matrix = water_background_data[:, 1:]  # Exclude first column (excitation wavelengths)

# Trim DLF matrix to match the water background matrix size
num_columns_bg = water_background_matrix.shape[1]  # Get number of columns in water background
dlf_matrix = dlf_matrix[:, :num_columns_bg]  # Trim DLF matrix to the same width

# Trim the x-axis labels to match the reduced matrix
wavelengths_x = wavelengths_x_full[:num_columns_bg]

# Subtract water background from DLF matrix
corrected_matrix = dlf_matrix - water_background_matrix

# **Load Calibration Data**
calib_df = pd.read_csv(calib_file, delimiter=';', skiprows=2)  # Skip header lines
calib_wavelengths = calib_df.iloc[:, 0].values  # First column (wavelengths)
calib_counts = calib_df.iloc[:, 1].values  # Second column (calibration counts)

# **Interpolate calibration values to match our x-axis wavelengths**
calib_interp = interp1d(calib_wavelengths, calib_counts, kind='linear', fill_value="extrapolate")
calib_values = calib_interp(wavelengths_x)  # Get calibration values for our x-axis

# **Apply calibration correction**
calibrated_matrix = calib_values / corrected_matrix  # Element-wise division

# **Invert the x and y axes**
calibrated_matrix = calibrated_matrix.T  # Transpose the matrix to swap axes
wavelengths_x, wavelengths_y = wavelengths_y, wavelengths_x  # Swap axis labels

# Round x-axis labels for better readability
wavelengths_x = np.round(wavelengths_x).astype(int)

# **Plot heatmap**
plt.figure(figsize=(12, 8))
sns.heatmap(calibrated_matrix, xticklabels=wavelengths_x, yticklabels=wavelengths_y, cmap='viridis', cbar=True)

# Adjust label spacing
plt.xticks(ticks=np.arange(0, len(wavelengths_x), step=10), labels=wavelengths_x[::10], rotation=90)
plt.yticks(ticks=np.arange(0, len(wavelengths_y), step=10), labels=wavelengths_y[::10], rotation=0)

# Invert the y-axis so that it starts from the lowest excitation wavelength
plt.gca().invert_yaxis()

# Label the axes (swapped)
plt.xlabel('Excitation Wavelength (nm)')  # Was Emission before
plt.ylabel('Emission Wavelength (nm)')  # Was Excitation before
plt.title('Calibrated 2D Heatmap (DLF - Water Background)')

# Show the plot
plt.tight_layout()
plt.show()
