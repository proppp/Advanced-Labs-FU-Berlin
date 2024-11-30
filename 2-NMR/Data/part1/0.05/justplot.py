
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Folder containing .dat files
folder_path = "."
prominence = 0.1  # Set the prominence for peak detection

plt.figure(figsize=(10, 6))  # Create a larger figure for better visualization

# Iterate through all .dat files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".dat"):
        file_path = os.path.join(folder_path, filename)

        # Load the data (assume tab-separated and in scientific notation)
        data = np.loadtxt(file_path)

        # Extract columns
        x = data[:, 0]  # First column as x-axis
        y = data[:, 1]  # Second column as y-axis

        # Find peaks with the specified prominence
        peaks, properties = find_peaks(y, prominence=prominence)

        if peaks.size > 0:
            # Get the position of the first peak
            first_peak_position = x[peaks[0]]

            # Adjust the x-axis to set the first peak's position to zero
            x_aligned = x - first_peak_position

            # Plot the data with adjusted x-axis
            plt.plot(x_aligned, y, label=filename)
        else:
            print(f"No peak with prominence {prominence} found in {filename}")

# Customize the plot
plt.xlabel("Adjusted Time (s)")
plt.ylabel("Amplitude")
plt.title("Aligned Plots from .dat files")
plt.legend()  # Show the legend
plt.grid(True)
plt.show()
