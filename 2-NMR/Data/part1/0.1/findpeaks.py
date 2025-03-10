
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# Folder containing .dat files
folder_path = "."

# Calibration parameters
calibration_prominence = 0.1  # Prominence for finding the first peak to align the data

# Peak marking parameters
peak_prominence = 0.02  # You can adjust this value for marking peaks

# New exponential fit function: A * e^(-t/T_2)
def exp_decay(t, A, T_2):
    return A * np.exp(-t / T_2)

# Plot settings
plt.figure(figsize=(10, 6))  # Create a larger figure for better visualization

# Arrays to store global peaks across files
global_x = []
global_y = []

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

            # Find all peaks with the specified prominence (for marking purposes)
            peaks_marked, _ = find_peaks(y_aligned, prominence=peak_prominence)

            # Filter out peaks with y < 0.1 AND x < 0.02
            valid_peaks = [
                i for i in peaks_marked
                if not ((y_aligned[i] < 0.1 and x_aligned[i] < 0.015) or (y_aligned[i] < 0.35 and x_aligned[i] < 0.004))
            ]

            # Collect valid peaks into global arrays
            global_x.extend(x_aligned[valid_peaks])
            global_y.extend(y_aligned[valid_peaks])

            # Plot the aligned data with reduced alpha for fading
            plt.plot(x_aligned, y_aligned, color='blue', alpha=0.3)

            # Mark the valid peaks with 'x' on the plot
            plt.scatter(x_aligned[valid_peaks], y_aligned[valid_peaks], color='red', marker='x')
        else:
            print(f"No peak with prominence {calibration_prominence} found in {filename}")

# Perform an exponential fit on the combined peaks
if len(global_x) >= 3:  # Ensure enough points for fitting
    global_x = np.array(global_x)
    global_y = np.array(global_y)

    # Perform the curve fit with the new fitting function
    popt, pcov = curve_fit(exp_decay, global_x, global_y, p0=(0.5, 0.001))

    # Calculate the standard deviations (errors) of the fit parameters
    perr = np.sqrt(np.diag(pcov))  # Standard errors from the covariance matrix

    # Generate smooth x values for plotting the fit
    x_fit = np.linspace(min(global_x), max(global_x), 500)
    y_fit = exp_decay(x_fit, *popt)

    # Plot the exponential fit
    plt.plot(x_fit, y_fit, color='black', label=f'Exp Fit: A={popt[0]:.4f}±{perr[0]:.4f}, T_2={popt[1]:.4f}±{perr[1]:.4f}')
    print(f"Fitted parameters: A={popt[0]:.4f} ± {perr[0]:.4f}, T_2={popt[1]:.4f} ± {perr[1]:.4f}")
else:
    print("Not enough points for fitting across all files.")

# Customize the plot
plt.xlabel("Adjusted Time (s)")
plt.ylabel("Amplitude")
plt.title("Aligned Plots with Exponential Fits")
plt.xlim(-0.003, 0.034)

plt.title("Spin-Echo Sequences for CuSO₄, 0.1M (Hahn Echo Method)")
plt.legend()
plt.grid(True)

# Save the plot as a PDF
output_pdf_path = "Spin_echo_0.1M.pdf"
plt.savefig(output_pdf_path, format='pdf')

# Show the plot
plt.show()

print(f"Plot saved as {output_pdf_path}")
