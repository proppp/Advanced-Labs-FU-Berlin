
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# Folder containing .dat files
folder_path = "."
# Calibration parameters
calibration_prominence = 0.1  # Prominence for finding the first peak to align the data

# Peak marking parameters
peak_prominence = 0.01  # You can adjust this value for marking peaks

# Exponential fit function (A * exp(-t / T0))
def exp_decay(t, A, T0):
    return A * np.exp(-t / T0)

# Plot settings
plt.figure(figsize=(10, 6))  # Create a larger figure for better visualization

# List to store aligned data
all_x = []
all_y = []

# Iterate through all .dat files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".dat"):
        file_path = os.path.join(folder_path, filename)

        # Load the data (assume tab-separated and in scientific notation)
        data = np.loadtxt(file_path)

        # Extract columns
        x = data[:, 0]  # First column as x-axis
        y = data[:, 1]  # Second column as y-axis

        # Find the first peak to align the data with the specified calibration prominence
        peaks, properties = find_peaks(y, prominence=calibration_prominence)

        if peaks.size > 0:
            # Get the position of the first peak
            first_peak_position = x[peaks[0]]

            # Adjust the x-axis to set the first peak's position to zero
            x_aligned = x - first_peak_position

            # Restrict the x-axis to the range [-0.002, max(x_aligned)]
            x_min = -0.002
            x_max = x_aligned[-1]
            mask = (x_aligned >= x_min) & (x_aligned <= x_max)
            x_aligned = x_aligned[mask]
            y_aligned = y[mask]

            # Add this data to the list
            all_x.append(x_aligned)
            all_y.append(y_aligned)

            # Plot the aligned data with a very low alpha for fading
            plt.plot(x_aligned, y_aligned, color='blue', alpha=0.1)  # More faded

        else:
            print(f"No peak with prominence {calibration_prominence} found in {filename}")

# If we have data, interpolate to a common x range
if len(all_x) > 0:
    # Define a common x range (you could adjust this range if needed)
    common_x = np.linspace(min(all_x[0]), max(all_x[0]), 500)

    # Interpolate all y data to the common x range
    interpolated_y = []
    for x_aligned, y_aligned in zip(all_x, all_y):
        # Interpolate each y to the common x
        interpolator = interp1d(x_aligned, y_aligned, kind='linear', fill_value="extrapolate")
        interpolated_y.append(interpolator(common_x))

    # Convert the list to a 2D array
    interpolated_y = np.array(interpolated_y)

    # Calculate the average of the interpolated y values
    avg_y = np.mean(interpolated_y, axis=0)

    # Plot the average data with a prominent green line
    plt.plot(common_x, avg_y, color='green', label='Average', linewidth=3)  # More prominent

    # Now fit an exponential decay to the green line within the range 0 to 0.0002
    fit_range_mask = (common_x >= 0) & (common_x <= 0.00025)
    x_fit = common_x[fit_range_mask]
    y_fit = avg_y[fit_range_mask]

    # Perform the exponential fit
    try:
        popt, pcov = curve_fit(exp_decay, x_fit, y_fit, p0=(1, 0.001))
        A, T0 = popt
        perr = np.sqrt(np.diag(pcov))  # Calculate the standard deviation (uncertainty) from the covariance matrix
        A_err, T0_err = perr

        # Generate the fitted y values for the fit range
        y_fitted = exp_decay(x_fit, *popt)

        # Plot the fitted exponential curve
        plt.plot(x_fit, y_fitted, color='red', label=f'Exp Fit: A={A:.3f}±{A_err:.3f}, T0={T0:.5f}±{T0_err:.5f}', linestyle='--', linewidth=2)
    except RuntimeError:
        print("Could not fit an exponential to the data.")

# Customize the plot
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Extract concentration from the filename (assuming filename format like '0.1-0.0001s.dat')
concentration = filename.split('-')[0]  # This assumes the concentration is before the first '-'
plt.title(f"Free Induction Decay for CuSO₄, 0.05M")

# Set the x and y limits as requested
plt.xlim(-0.0001, 0.0005)  # Restrict the x-axis to the fitting range
plt.ylim(0, 0.35)  # Adjust y-axis to fit the selected data

# Show the legend
plt.legend()

# Grid and show the plot
plt.grid(True)

# Save the plot as a PDF
plt.savefig(f"Free_Induction_Decay_0.05M.pdf")

# Display the plot
plt.show()
