
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the .csv file
file_path = 'CNT.csv'  # Change this to your actual file path
file_path = 'WaterBackground.csv'

# Read the entire data into a numpy array
data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

# Extract x and y labels (wavelengths)
wavelengths_x = np.genfromtxt(file_path, delimiter=',', max_rows=1)[1:]  # Emission wavelengths (first row)
wavelengths_y = np.genfromtxt(file_path, delimiter=',', skip_header=1, usecols=0)  # Excitation wavelengths (first column)

# Convert the data into a matrix (excluding row and column headers)
data_matrix = data[:, 1:]  # Remove the first column (excitation wavelengths)

# Invert x and y axes
data_matrix = data_matrix.T  # Transpose the matrix to swap axes
wavelengths_x, wavelengths_y = wavelengths_y, wavelengths_x  # Swap the labels

# Round x-axis labels for better readability
wavelengths_x = np.round(wavelengths_x).astype(int)

# Print the first few rows of the matrix
print("Data Matrix (First 5 Rows & Columns):")
print(data_matrix[:5, :5])

# Create the 2D heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data_matrix, xticklabels=wavelengths_x, yticklabels=wavelengths_y, cmap='viridis', cbar=True)

# Adjust label spacing for clarity
plt.xticks(ticks=np.arange(0, len(wavelengths_x), step=10), labels=wavelengths_x[::10], rotation=90)
plt.yticks(ticks=np.arange(0, len(wavelengths_y), step=10), labels=wavelengths_y[::10], rotation=0)

# Invert the y-axis so that it starts from the lowest excitation wavelength
plt.gca().invert_yaxis()

# Label the axes
plt.xlabel('Emission Wavelength (nm)')
plt.ylabel('Excitation Wavelength (nm)')
plt.title('2D Heatmap of Count Intensity')

# Show the plot
plt.tight_layout()
plt.show()
