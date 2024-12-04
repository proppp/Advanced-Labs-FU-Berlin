
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
x_threshold = 0.014
# Desired x range
x_min = -0.007
x_max = 0.1  # Adjust the upper limit based on your data range

# List to store green peaks (second peaks)
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
        else:
            # If fewer than 2 peaks are found, only shift x
            peak_indices = peaks[np.argsort(x[peaks])]
            x_shifted = x - x[peak_indices[0]] if len(peak_indices) > 0 else x

        # Plot the adjusted data with blue color and alpha 0.5
        plt.plot(x_shifted, y, color='blue', alpha=0.5)

        # Mark only the green peaks (second peak)
        for i, peak_index in enumerate(peak_indices[1:]):  # Start from the second peak
            # Save the green peak x and y values for fitting
            green_peaks_x.append(x_shifted[peak_index])
            green_peaks_y.append(y[peak_index])
            plt.scatter(x_shifted[peak_index], y[peak_index], color='g')

# Convert the green peaks lists to numpy arrays for fitting
green_peaks_x = np.array(green_peaks_x)
green_peaks_y = np.array(green_peaks_y)

# Define the function to fit: A(t) = A(1 - 2e^(-t/T_1))
def fit_function(x, A, T_1):
    return A * (1 - 2 * np.exp(-x / T_1))

# Attempt to fit the function to the collected green peaks
try:
    # Fit the curve and obtain the parameters and covariance matrix
    popt, pcov = curve_fit(fit_function, green_peaks_x, green_peaks_y, p0=[1, 0.01], maxfev=10000)

    # Extract fit parameters
    A, T_1 = popt

    # Compute the fitting uncertainties (standard deviations)
    A_error, T_1_error = np.sqrt(np.diag(pcov))  # Diagonal elements of covariance matrix

    # Compute the fitted curve values for the data points
    y_fit = fit_function(green_peaks_x, *popt)

    # Compute the residuals (differences between fitted and actual values)
    residuals = green_peaks_y - y_fit

    # Compute the Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean(residuals**2))

    # Generate the fitted curve for plotting
    x_fit = np.linspace(min(green_peaks_x), max(green_peaks_x), 1000)
    y_fit_curve = fit_function(x_fit, *popt)

    # Create the label with fit parameters, uncertainties, and RMSE
    label_text = f"Fit: A={A:.3f} ± {A_error:.3f}, T_1={T_1:.3f} ± {T_1_error:.3f}"

    # Plot the fitted function
    plt.plot(x_fit, y_fit_curve, color='purple', label=label_text)

except Exception as e:
    print(f"Fit did not converge or failed: {e}")
    # If fitting fails, skip plotting the fitted curve but still show the data

# Set the x-axis limits to the specified range
plt.xlim(x_min, x_max)

# Customize the plot
# Customize the plot
plt.xlabel("Adjusted time (Peak 1 at x=0)")
plt.ylabel("Amplitude (a.u.)")
plt.title("CuSO₄ 0.05M Inversion Recovery Method")
plt.grid(True)

# Show the legend with the fit parameters
plt.legend()

# Save the plot as a PDF
plt.savefig("IR_0.05.pdf", format='pdf')

# Show the plot
plt.show()
