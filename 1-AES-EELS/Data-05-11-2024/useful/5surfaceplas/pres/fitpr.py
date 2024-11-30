
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
    {"name": "Excitation", "initial_mu": 185.5, "range": (177, 194)},
    {"name": "B1", "initial_mu": 166, "range": (162, 170)},
    {"name": "B2", "initial_mu": 148.5, "range": (143, 154)},
    {"name": "B3", "initial_mu": 134.5, "range": (129, 140)},
    {"name": "S1", "initial_mu": 124.5, "range": (122, 127)},
    {"name": "S2", "initial_mu": 114, "range": (110, 118)}
]

# Shift value to add to all energy values (15.74 eV)
shift_value = 15.74

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

        # First Plot (Without Gaussian Fits) - Displaying only the raw data
        plt.figure(figsize=(10, 6))
        plt.plot(df['X'], df['Y'], linestyle='-', color='b', label='Data')
        plt.title("Energy Spectrum (Without Gaussian Fits)", fontsize=16)
        plt.xlabel("Corrected Energy (eV)", fontsize=14)
        plt.ylabel("dN/dE (a.u.)", fontsize=14)
        plt.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)

        # Save first plot as PNG for Google Docs compatibility
        output_file_path_no_fit = file_path.replace('.txt', '_no_fit_plot.png')
        plt.savefig(output_file_path_no_fit, format='png', dpi=300)
        print(f"Plot without Gaussian fits saved as PNG: {output_file_path_no_fit}")
        plt.close()  # Close to free up memory

        # Second Plot (With Gaussian Fits) - Displaying data with Gaussian derivative fits
        plt.figure(figsize=(10, 6))
        plt.plot(df['X'], df['Y'], linestyle='-', color='b', label='Data')
        plt.title("Energy Spectrum with Gaussian Derivative Fits", fontsize=16)
        plt.xlabel("Corrected Energy (eV)", fontsize=14)
        plt.ylabel("dN/dE (a.u.)", fontsize=14)
        plt.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

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

        plt.legend(fontsize=12)

        # Save second plot as PNG for Google Docs compatibility
        output_file_path_with_fit = file_path.replace('.txt', '_fit_plot.png')
        plt.savefig(output_file_path_with_fit, format='png', dpi=300)
        print(f"Plot with Gaussian fits saved as PNG: {output_file_path_with_fit}")
        plt.close()  # Close to free up memory

    else:
        print(f"No valid data found in {file_path}.")

# Example usage:
file_path = './surfaceplasmons_30uV_156deg_300ms_1.9e-09mbar_200eV.txt'  # Replace with your actual file path
fit_al_kll_peaks_with_baseline(file_path)
