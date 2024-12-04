
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# Folder containing .dat files
folder_path = "."
# User-defined prominence value
prominence_value = 0.02
# Threshold for deciding inversion
x_threshold = 0.00373
# Desired x range
x_min = -0.007
x_max = 0.1  # Adjust the upper limit based on your data range

# Logarithmic function to fit
def log_func(x, a, b, c, d):
    return a * np.log(b * x + c) + d

# Arrays to store the x and y positions of the green peaks
green_peaks_x = []
green_peaks_y = []

# Iterate through all .dat files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".dat"):
        file_path = os.path.join(folder_path, filename)

        # Load the data (assume tab-separated and in scientific notation)
        data = np.loadtxt(file_path)

        # Extract columns
        x = data[:, 0]  # First column as x-axis
        y = data[:, 1]  # Second column as y-axis

        # Normalize by subtracting the average y value
        y_mean = np.mean(y)
        y = y - y_mean

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

            # Check condition for inversion
            if x_shifted[peak_indices[1]] < x_threshold:
                y = -y  # Invert the graph
        elif len(peaks) == 1:
            # Handle case where only 1 peak is detected
            peak_indices = peaks
            x_shifted = x - x[peak_indices[0]]  # Only shift using the first peak
        else:
            # Skip files with no peaks
            continue

        # Save the second (green) peak x and y values for fitting (only if 2 peaks are found)
        if len(peak_indices) > 1:
            green_peaks_x.append(x_shifted[peak_indices[1]])  # Second peak (green) x value
            green_peaks_y.append(y[peak_indices[1]])  # Second peak (green) y value
        elif len(peak_indices) == 1:
            # If only one peak, append it as green peak for fitting
            green_peaks_x.append(x_shifted[peak_indices[0]])  # Only one peak
            green_peaks_y.append(y[peak_indices[0]])  # Only one peak

        # Plot the adjusted data with blue color and alpha 0.5
        plt.plot(x_shifted, y, color='blue', alpha=0.5)

        # Mark only the green peaks (second peak or first if only one detected)
        plt.scatter(x_shifted[peak_indices[1] if len(peak_indices) > 1 else 0], y[peak_indices[1] if len(peak_indices) > 1 else 0], color='g')

# Set the x-axis limits to the specified range
plt.xlim(x_min, x_max)

# Customize the plot
plt.xlabel("Shifted X-axis (Peak 1 at x=0)")
plt.ylabel("Y-axis")
plt.title("Shifted, Adjusted, and Normalized Plots with Green Peaks")
plt.grid(True)
plt.show()

# Convert the green peaks lists to arrays for fitting
green_peaks_x = np.array(green_peaks_x)
green_peaks_y = np.array(green_peaks_y)

# Fit the logarithmic function to the green peaks
params, covariance = curve_fit(log_func, green_peaks_x, green_peaks_y, p0=[1, 1, 1, 1])

# Plot the green peaks and the logarithmic fit
plt.scatter(green_peaks_x, green_peaks_y, color='g', label='Green Peaks')
x_fit = np.linspace(min(green_peaks_x), max(green_peaks_x), 1000)
y_fit = log_func(x_fit, *params)

plt.plot(x_fit, y_fit, color='r', label="Logarithmic Fit")
plt.xlabel("X (Shifted)")
plt.ylabel("Y")
plt.title("Logarithmic Fit to Green Peaks")
plt.legend()
plt.grid(True)
plt.show()
