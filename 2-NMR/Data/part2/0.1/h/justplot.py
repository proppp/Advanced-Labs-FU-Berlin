
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Folder containing .dat files
folder_path = "."

# Iterate through all .dat files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".dat"):
        file_path = os.path.join(folder_path, filename)

        # Load the data (assume tab-separated and in scientific notation)
        data = np.loadtxt(file_path)

        # Extract columns
        x = data[:, 0]  # First column as x-axis
        y = data[:, 1]  # Second column as y-axis

        # Find peaks
        peaks, _ = find_peaks(y)  # Identify peaks in y
        if len(peaks) >= 2:
            # Get the indices of the two highest peaks
            highest_peaks = np.argsort(y[peaks])[-2:]  # Top two peak indices
            peak_indices = peaks[highest_peaks]       # Corresponding indices in the data
        else:
            # If fewer than 2 peaks are found, use all available
            peak_indices = peaks

        # Plot the data
        plt.plot(x, y, label=filename)

        # Mark the peaks
        colors = ['r', 'g']  # Different colors for the two peaks
        for i, peak_index in enumerate(peak_indices):
            plt.scatter(x[peak_index], y[peak_index], color=colors[i % len(colors)],
                        label=f"Peak {i+1} in {filename}")

# Customize the plot
plt.xlabel("X-axis Label")
plt.ylabel("Y-axis Label")
plt.title("Plots with Peaks from .dat Files")
plt.legend()  # Show the legend
plt.grid(True)
plt.show()
