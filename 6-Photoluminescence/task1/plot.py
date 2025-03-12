
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the .txt file
file_path = 'data.txt'  # Change this to your actual file path

file_path = 'Dflt3d.csv'  # Make sure to change this to the correct path of your text file
# Read the data into a numpy array
data = np.genfromtxt(file_path, delimiter=';', skip_header=1)

# Extract the x and y labels (wavelengths) from the first row and column
wavelengths_x = np.genfromtxt(file_path, delimiter=';', max_rows=1)[1:]  # The first row is for x-axis (Emission)
wavelengths_y = np.genfromtxt(file_path, delimiter=';', skip_header=1, usecols=0)  # First column is for y-axis (Excitation)

# Invert the x and y axes
data = data.T  # Transpose the data to swap rows and columns
wavelengths_x, wavelengths_y = wavelengths_y, wavelengths_x  # Swap the axis labels

# Round x-axis labels to whole numbers
wavelengths_x = np.round(wavelengths_x).astype(int)

# Create the 2D heatmap plot
plt.figure(figsize=(12, 8))
sns.heatmap(data, xticklabels=wavelengths_x, yticklabels=wavelengths_y, cmap='viridis', cbar=True)

# Adjust label spacing by limiting the number of ticks
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
