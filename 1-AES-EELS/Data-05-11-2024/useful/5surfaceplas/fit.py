
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Define the Gaussian derivative function with a baseline
def gaussian_derivative_with_baseline(x, A, mu, sigma, C):
    return A * (-2 * (x - mu) / sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2)) + C

# Updated peak information with the new ranges
peak_info = [
    {"name": "Excitation", "initial_mu": 205, "range": (200, 211)},
    {"name": "S1", "initial_mu": 194, "range": (192, 196)},
    {"name": "B1", "initial_mu": 187, "range": (183, 189)},
    {"name": "B2", "initial_mu": 171, "range": (165, 174)},
    {"name": "B3", "initial_mu": 153, "range": (150, 156)},
    {"name": "B4", "initial_mu": 134.5, "range": (135, 140)},  # Kept as it is
]
# Shift value to add to all energy values (15.74 eV)
shift_value = 0

# Define the assumed error in X (e.g., 1.5 eV)
error_x = 1.5

def fit_al_kll_peaks_with_baseline(file_path):
    # Read the file and skip the header
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Read the data section
    data = []
    for line in lines[6:]:  # Start from the 7th line onward, assuming header ends at line 6
        if line.strip():  # Ensure line is not empty
            parts = line.split()
            if len(parts) == 2:  # We expect two values per line
                try:
                    x_value = float(parts[0]) + shift_value  # Add shift to energy
                    y_value = float(parts[1])
                    data.append((x_value, y_value))
                except ValueError:
                    print(f"Skipping line due to conversion error: {line}")

    # Convert the data to a DataFrame for easier plotting
    if data:
        df = pd.DataFrame(data, columns=['X', 'Y'])

        # Plot the original data as a continuous line
        plt.figure(figsize=(10, 6))
        plt.plot(df['X'], df['Y'], linestyle='-', color='b', label='Data')
        plt.title("Data and Gaussian Derivative Fits for Al Surface and Bulk Plasmons")
        plt.xlabel("Corrected Energy (eV)")
        plt.ylabel("dN/dE (a.u.)")
        #plt.grid()

        # Fit each peak in the specified ranges and collect `mu` values
        colors = ['r', 'g', 'purple', 'orange', 'cyan', 'magenta']  # Colors for each peak
        mu_values = []  # Store the fitted `mu` values for distance calculations

        for i, peak in enumerate(peak_info):
            # Adjust the peak range by adding the shift value
            adjusted_range = (peak['range'][0] + shift_value, peak['range'][1] + shift_value)

            # Filter data for the current peak range
            df_range = df[(df['X'] >= adjusted_range[0]) & (df['X'] <= adjusted_range[1])]

            # Fit the Gaussian derivative to the data in the specified range
            if not df_range.empty:
                try:
                    # Set initial guesses for fitting parameters
                    initial_A = 500  # Maximal expected height
                    initial_mu = peak['initial_mu'] + shift_value  # Add shift to initial_mu
                    initial_sigma = 10  # Adjusted guess for narrower peaks
                    initial_C = df_range['Y'].mean()  # Initial guess for baseline

                    # Weights for the fit (assuming constant error for all x-values)
                    weights = 1 / (error_x**2 * np.ones_like(df_range['X']))  # Uniform weights

                    # Fit the Gaussian derivative function with baseline, applying the weights
                    popt, pcov = curve_fit(
                        gaussian_derivative_with_baseline,
                        df_range['X'],
                        df_range['Y'],
                        p0=[initial_A, initial_mu, initial_sigma, initial_C],
                        sigma=weights  # Incorporate weights here
                    )

                    # Extract fitted parameters
                    A, mu, sigma, C = popt
                    mu_values.append(mu)  # Save the fitted `mu` for distance calculation

                    # Print the fitting parameters for the current peak
                    print(f"\nFitted Parameters for {peak['name']}:")
                    print(f"A = {A:.2f} ± {np.sqrt(pcov[0, 0]):.2f}")
                    print(f"mu = {mu:.2f} ± {np.sqrt(pcov[1, 1]):.2f}")
                    print(f"sigma = {sigma:.2f} ± {np.sqrt(pcov[2, 2]):.2f}")
                    print(f"C = {C:.2f} ± {np.sqrt(pcov[3, 3]):.2f}")

                    # Generate the fit curve
                    fit_curve = gaussian_derivative_with_baseline(df_range['X'], *popt)

                    # Plot the fit curve with a unique color
                    plt.plot(df_range['X'], fit_curve, linestyle='-', color=colors[i % len(colors)], label=peak['name'])

                    # Draw a vertical line at `mu` (no label)
                    plt.axvline(mu, color=colors[i % len(colors)], linestyle='--', alpha=0.7)

                except RuntimeError as e:
                    print(f"Error fitting Gaussian derivative for peak {peak['name']}: {e}")

        # Calculate and print distances between consecutive peaks
        print("\nDistances between consecutive peaks:")
        for j in range(1, len(mu_values)):
            distance = abs(mu_values[j] - mu_values[j - 1])
            print(f"Distance between {peak_info[j-1]['name']} and {peak_info[j]['name']}: {distance:.2f} eV")

            # Draw arrows between consecutive mu values
            plt.annotate('', xy=(mu_values[j], 10000), xytext=(mu_values[j - 1], 10000),
                         arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))

            # Annotate the distance along the arrow
            mid_point = (mu_values[j] + mu_values[j - 1]) / 2
            vertical_offset = df['Y'].max() * (0.75 - 0.05 * (j % 2))  # Alternating offsets for clarity
            plt.text(mid_point, vertical_offset, f"{distance:.2f} eV", color='black', ha='center', va='bottom')

        plt.legend()  # Ensure the legend is added after the peaks are plotted

        # Save the plot as a PDF file
        output_file_path = file_path.replace('.txt', '_Al_KLL_peaks_fit_with_baseline_plot_shifted.pdf')
        plt.savefig(output_file_path, format='pdf')
        plt.show()
        print(f"Plot saved as PDF: {output_file_path}")

        plt.close()  # Close the plot to free up memory

    else:
        print(f"No valid data found in {file_path}.")

# Example usage:
file_path = './surface3.txt'  # Replace with your actual file path
fit_al_kll_peaks_with_baseline(file_path)
