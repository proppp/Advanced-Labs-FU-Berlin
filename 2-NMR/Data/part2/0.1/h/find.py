
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Folder containing .dat files
folder_path = "."
# User-defined prominence value
prominence_value = float(input("Enter the desired prominence value: "))

# Threshold for second peak x-coordinate
x_threshold = 0.0144

# Iterate through all .dat files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".dat"):
        file_path = os.path.join(folder_path, filename)

        # Load the data (assume tab-separated and in scientific notation)
        data = np.loadtxt(file_path)

        # Extract columns
        x = data[:, 0]  # First column as x-axis
        y = data[:, 1]  # Second column as y-axis

        # Find peaks with specified prominence
        peaks, _ = find_peaks(y, prominence=prominence_value)  # Filter peaks by prominence
        if len(peaks) >= 2:
            # Get the indices of the two highest peaks
            highest_peaks = np.argsort(y[peaks])[-2:]  # Top two peak indices
            peak_indices = peaks[highest_peaks]       # Corresponding indices in the data

            # Ensure the first peak has the smaller x value
            peak_indices = peak_indices[np.argsort(x[peak_indices])]

            # Find x-coordinate of Peak 1 and shift x values
            x_shifted = x - x[peak_indices[0]]

            # Check if the second peak's x-coordinate is less than the threshold
            if x[peak_indices[1]] < x_threshold:
                y = -y  # Invert y-values
        else:
            # If fewer than 2 peaks are found, use all available and sort by x
            peak_indices = peaks[np.argsort(x[peaks])]
            x_shifted = x - x[peak_indices[0]] if len(peak_indices) > 0 else x

        # Plot the shifted data
        plt.plot(x_shifted, y, label=filename)

        # Mark the peaks on the shifted plot
        colors = ['r', 'g']  # Different colors for the two peaks
        for i, peak_index in enumerate(peak_indices):
            plt.scatter(x_shifted[peak_index], y[peak_index], color=colors[i % len(colors)],
                        label=f"Peak {i+1} in {filename}")

# Customize the plot
plt.xlabel("Shifted X-axis (Peak 1 at x=0)")
plt.ylabel("Y-axis (Inverted if Second Peak < Threshold)")
plt.title("Shifted and Inverted Plots with Peaks from .dat Files")
plt.legend()  # Show the legend
plt.grid(True)
plt.show()
