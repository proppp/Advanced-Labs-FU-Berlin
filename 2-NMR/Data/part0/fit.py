
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# Define the exponential function
def exponential_decay(t, A, T0):
    return A * np.exp(-t / T0)

# List of files, their respective fitting ranges, and plotting ranges
files_and_ranges = {
    "0.1-0.0001s.dat": {
        "fit_range": (0.0001791, 0.0004044),  # Fitting range for file 1
        "plot_range": (-0.0001, 0.001),      # Plotting range for file 1
    },
    "0.2-0.0001s.dat": {
        "fit_range": (0.0001489, 0.000601),  # Fitting range for file 2
        "plot_range": (-0.0002, 0.0012),     # Plotting range for file 2
    },
}

# Prominence for peak detection
prominence = 0.1

# Iterate through specified files and their ranges
for filename, ranges in files_and_ranges.items():
    file_path = os.path.join(".", filename)  # File is in the current folder

    if os.path.exists(file_path):  # Ensure the file exists
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

            # Select data within the specified fitting range
            fit_range_start, fit_range_end = ranges["fit_range"]
            mask = (x_aligned >= fit_range_start) & (x_aligned <= fit_range_end)
            x_fit = x_aligned[mask]
            y_fit = y[mask]

            plt.figure(figsize=(10, 6))  # Create a new figure for each file

            if len(x_fit) > 0:
                # Fit the exponential function to the selected data
                try:
                    popt, pcov = curve_fit(exponential_decay, x_fit, y_fit, p0=(1, 0.001))
                    A, T0 = popt
                    y_fitted = exponential_decay(x_fit, *popt)

                    # Calculate uncertainties (square root of diagonal of covariance matrix)
                    perr = np.sqrt(np.diag(pcov))
                    A_err, T0_err = perr

                    # Extract concentration from the filename
                    concentration = filename.split("-")[0]

                    # Plot the original data
                    plt.plot(x_aligned, y, label="Data")

                    # Overlay the fitted curve
                    plt.plot(
                        x_fit,
                        y_fitted,
                        '--',
                        label=f"Fit: A={A:.3f}±{A_err:.3f}, T0={T0:.5f}±{T0_err:.5f}"
                    )

                    # Add labels and title
                    plt.xlabel("Time (s)")
                    plt.ylabel("Amplitude")
                    plt.title(f"Free Induction Decay for CuSO₄, {concentration}M")
                    plt.legend()
                    plt.grid(True)

                    # Set custom plotting range
                    plot_range_start, plot_range_end = ranges["plot_range"]
                    plt.xlim(plot_range_start, plot_range_end)
                    plt.ylim(min(y) * 0.9, max(y) * 1.1)

                    # Save the plot as a PDF
                    output_filename = f"{filename.split('.')[1]}_fit.pdf"
                    plt.savefig(output_filename, format="pdf")
                    print(f"Plot saved as {output_filename}")

                except RuntimeError:
                    print(f"Could not fit data in {filename}")
            else:
                print(f"No data in range {fit_range_start}-{fit_range_end} for {filename}")

            # Show the plot for the current file
            plt.show()
        else:
            print(f"No peak with prominence {prominence} found in {filename}")
    else:
        print(f"File {filename} does not exist!")
