
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Define the Gaussian derivative function with a baseline
def gaussian_derivative_with_baseline(x, A, mu, sigma, C):
    return A * (-2 * (x - mu) / sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2)) + C

# Updated peak information with non-overlapping ranges (initial_mu is the unshifted energy)
peak_info = [
    {"name": "B3", "initial_mu": 973, "range": (973, 984)},
    {"name": "B2", "initial_mu": 986, "range": (986, 997)},
    {"name": "B1", "initial_mu": 997, "range": (997, 1013)},
    {"name": "Excitation", "initial_mu": 1013, "range": (1013, 1029)}
]

# Updated peak information with non-overlapping ranges (initial_mu is the unshifted energy)
peak_info = [
    {"name": "B6", "initial_mu": 922, "range": (922, 930)},
    {"name": "B5", "initial_mu": 938, "range": (938, 948)},
    {"name": "B4", "initial_mu": 953, "range": (953, 963)},
    {"name": "B3", "initial_mu": 973, "range": (967, 978)},
    {"name": "B2", "initial_mu": 986, "range": (981, 993)},
    {"name": "B1", "initial_mu": 997, "range": (997, 1010)},
    {"name": "Excitation", "initial_mu": 1013, "range": (1013, 1029)}
]



# Shift value to add to all energy values (15.74 eV)
shift_value = 15.74

# Define the assumed error in X (e.g., 0.5 eV)
error_x = 0.5

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

        # Plot 1: Original data without Gaussian derivative fits
        plt.figure(figsize=(10, 6))
        plt.plot(df['X'], df['Y'], linestyle='-', color='b', label='Data')
        plt.title("Energy Spectrum of Al Bulk Plasmons (Without Gaussian Fits)", fontsize=16)
        plt.xlabel("Corrected Energy (eV)", fontsize=14)
        plt.ylabel("dN/dE (a.u.)", fontsize=14)
        plt.grid()
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)

        # Save the plot as a PNG file
        output_file_path = file_path.replace('.txt', '_bulk_plasmons_no_fit_plot.png')
        plt.savefig(output_file_path, format='png', dpi=300)
        plt.show()
        print(f"Plot without Gaussian fits saved as PNG: {output_file_path}")

        plt.close()  # Close the plot to free up memory

        # Plot 2: Data with Gaussian derivative fits
        plt.figure(figsize=(10, 6))
        plt.plot(df['X'], df['Y'], linestyle='-', color='b', label='Data')
        plt.title("Energy Spectrum of Al Bulk Plasmons with Gaussian Derivative Fits", fontsize=16)
        plt.xlabel("Corrected Energy (eV)", fontsize=14)
        plt.ylabel("dN/dE (a.u.)", fontsize=14)
        plt.grid()
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # List to store fitted mu values
        mu_values = []

        # Colors for Gaussian derivative fits
        colors = ['r', 'g', 'purple', 'orange']  # Colors for each peak
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
                    initial_sigma = 50
                    initial_C = df_range['Y'].mean()  # Initial guess for baseline

                    # Weights for the fit (assuming constant error for all x-values)
                    weights = 1 / (error_x**2 * np.ones_like(df_range['X']))  # Uniform weights

                    # Fit the Gaussian derivative function with baseline, applying the weights
                    popt, pcov = curve_fit(
                        gaussian_derivative_with_baseline,
                        df_range['X'],
                        df_range['Y'],
                        p0=[initial_A, initial_mu, initial_sigma, initial_C],
                        sigma=weights
                    )

                    # Extract fitted parameters
                    A, mu, sigma, C = popt

                    # Store the fitted mu for distance calculation
                    mu_values.append(mu)

                    # Calculate the errors (standard deviation) of the fitted parameters
                    perr = np.sqrt(np.diag(pcov))

                    print(f"Fitted Parameters for {peak['name']}:")
                    print(f"A = {A:.2f} ± {perr[0]:.2f}")
                    print(f"mu = {mu:.2f} ± {perr[1]:.2f}")
                    print(f"sigma = {sigma:.2f} ± {perr[2]:.2f}")
                    print(f"C = {C:.2f} ± {perr[3]:.2f}")

                    # Generate the fit curve
                    fit_curve = gaussian_derivative_with_baseline(df_range['X'], *popt)

                    # Plot the fit curve with a unique color and label with fitted mu
                    plt.plot(df_range['X'], fit_curve, linestyle='-', color=colors[i % len(colors)],
                             label=f"{peak['name']} ({mu:.2f} eV)")

                    # Draw a vertical line at `mu` (peak position)
                    plt.axvline(mu, color=colors[i % len(colors)], linestyle='--', alpha=0.7)

                except RuntimeError as e:
                    print(f"Error fitting Gaussian derivative for peak {peak['name']}: {e}")

        # Compute and print distances between each subsequent mu
        for i in range(1, len(mu_values)):
            distance = mu_values[i] - mu_values[i - 1]
            print(f"Distance between mu for {peak_info[i-1]['name']} and {peak_info[i]['name']}: {distance:.2f} eV")

            # Draw arrows between consecutive mu values
            plt.annotate('', xy=(mu_values[i], -6000), xytext=(mu_values[i - 1], -6000),
                         arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))

            # Annotate the distance along the arrow
            mid_point = (mu_values[i] + mu_values[i - 1]) / 2
            vertical_offset = df['Y'].max() * (0.00001 - 0.9 )  # Alternating offsets for clarity
            plt.text(mid_point, vertical_offset, f"{distance:.2f} eV", color='black', ha='center', va='bottom')

        plt.legend(fontsize=12)

        # Save the plot as a PNG file
        output_file_path = file_path.replace('.txt', '_bulk_plasmons_with_fit_plot.png')
        plt.savefig(output_file_path, format='png', dpi=300)
        plt.show()
        print(f"Plot with Gaussian fits saved as PNG: {output_file_path}")

        plt.close()  # Close the plot to free up memory

    else:
        print(f"No valid data found in {file_path}.")

# Example usage:
#file_path = './bulkplasmons_300uV_156deg_300ms_1.9e-09mbar_1keV_1.5V_amplitude.txt'  # Replace with your actual file path

file_path = './bulk1.txt'
fit_al_kll_peaks_with_baseline(file_path)
