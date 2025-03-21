
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
cnt_file = 'CNT.csv'
water_bg_file = 'WaterBackground.csv'

# Load CNT data
cnt_data = np.genfromtxt(cnt_file, delimiter=',', skip_header=1)
wavelengths_x_cnt = np.genfromtxt(cnt_file, delimiter=',', max_rows=1)[1:]  # Emission wavelengths
wavelengths_y_cnt = np.genfromtxt(cnt_file, delimiter=',', skip_header=1, usecols=0)  # Excitation wavelengths
cnt_matrix = cnt_data[:, 1:]  # Remove first column (which contains excitation wavelengths)

# Load Water Background data
water_data = np.genfromtxt(water_bg_file, delimiter=',', skip_header=1)
wavelengths_x_water = np.genfromtxt(water_bg_file, delimiter=',', max_rows=1)[1:]  # Emission wavelengths for background
wavelengths_y_water = np.genfromtxt(water_bg_file, delimiter=',', skip_header=1, usecols=0)  # Excitation wavelengths for background
water_matrix = water_data[:, 1:]  # Remove first column (which contains excitation wavelengths)

# Find indices where wavelengths match
x_match_indices = [np.where(wavelengths_x_cnt == wx)[0][0] for wx in wavelengths_x_water if wx in wavelengths_x_cnt]
y_match_indices = [np.where(wavelengths_y_cnt == wy)[0][0] for wy in wavelengths_y_water if wy in wavelengths_y_cnt]

# Create an empty matrix of the same size as CNT, filled with zeros
expanded_water_matrix = np.zeros_like(cnt_matrix)

# Insert water background values into the correct positions
for i, y_idx in enumerate(y_match_indices):
    for j, x_idx in enumerate(x_match_indices):
        expanded_water_matrix[y_idx, x_idx] = water_matrix[i, j]

# Subtract the expanded water background from the CNT matrix
corrected_matrix = cnt_matrix - expanded_water_matrix

# Print first few values of corrected matrix
print("Corrected Data Matrix (First 5 Rows & Columns):")
print(corrected_matrix[:5, :5])

# Invert x and y axes
corrected_matrix = corrected_matrix.T  # Transpose matrix
wavelengths_x_cnt, wavelengths_y_cnt = wavelengths_y_cnt, wavelengths_x_cnt  # Swap axis labels

# Round x-axis labels for better readability
wavelengths_x_cnt = np.round(wavelengths_x_cnt).astype(int)

# Plot the corrected heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corrected_matrix, xticklabels=wavelengths_x_cnt, yticklabels=wavelengths_y_cnt, cmap='viridis', cbar=True)

# Adjust label spacing
plt.xticks(ticks=np.arange(0, len(wavelengths_x_cnt), step=10), labels=wavelengths_x_cnt[::10], rotation=90)
plt.yticks(ticks=np.arange(0, len(wavelengths_y_cnt), step=10), labels=wavelengths_y_cnt[::10], rotation=0)

# Invert y-axis for correct orientation
plt.gca().invert_yaxis()

# Label the axes
plt.xlabel('Emission Wavelength (nm)')
plt.ylabel('Excitation Wavelength (nm)')
plt.title('Corrected 2D Heatmap (CNT - Water Background)')

# Show the plot
plt.tight_layout()
plt.show()
