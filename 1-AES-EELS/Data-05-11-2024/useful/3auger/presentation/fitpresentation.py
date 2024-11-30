
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Define the Gaussian derivative function with a baseline
def gaussian_derivative_with_baseline(x, A, mu, sigma, C):
    return A * (-2 * (x - mu) / sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2)) + C

# Updated peak information with non-overlapping ranges
peak_info = [
    {"name": "K-L$_{2,3}$L$_{2,3}$ ($^{1}$D)", "initial_mu": 1393.3, "range": (1372, 1385)},
    {"name": "K-L$_{2,3}$L$_{2,3}$ ($^{1}$D) + B1", "initial_mu": 1386.85, "range": (1354, 1370)},
    {"name": "K-L$_{1}$L$_{2,3}$ ($^{3}$P)", "initial_mu": 1357.85, "range": (1338, 1351)},
    {"name": "K-L$_{1}$L$_{2,3}$ ($^{1}$P)", "initial_mu": 1341.95, "range": (1320, 1336)},
    {"name": "K-L$_{1}$L$_{2,3}$ ($^{1}$P) + B1", "initial_mu": 1303.3, "range": (1296, 1318)}
]

# Shift value to add to all energy values (15.74 eV)
shift_value = 15.74
error_x = 1.5

def fit_al_kll_peaks_with_baseline(file_path):
    # Read the file and skip the header
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Read the data section
    data = []
    for line in lines[6:]:  # Start from the 7th line onward, assuming header ends at line 6
        if line.strip():
            parts = line.split()
            if len(parts) == 2:
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
        plt.title("Energy Spectrum of Al KLL Transitions (Without Gaussian Fits)", fontsize=16)
        plt.xlabel("Corrected Energy (eV)", fontsize=14)
        plt.ylabel("dN/dE (a.u.)", fontsize=14)
        #plt.grid()
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)

        # Save the plot as a PNG file
        output_file_path = file_path.replace('.txt', '_Al_KLL_no_fit_plot.png')
        plt.savefig(output_file_path, format='png', dpi=300)
        plt.show()
        print(f"Plot without Gaussian fits saved as PNG: {output_file_path}")

        plt.close()  # Close the plot to free up memory

        # Plot 2: Data with Gaussian derivative fits
        plt.figure(figsize=(10, 6))
        plt.plot(df['X'], df['Y'], linestyle='-', color='b', label='Data')
        plt.title("Energy Spectrum of Al KLL Transitions with Gaussian Derivative Fits", fontsize=16)
        plt.xlabel("Corrected Energy (eV)", fontsize=14)
        plt.ylabel("dN/dE (a.u.)", fontsize=14)
        #plt.grid()
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Colors for Gaussian derivative fits
        colors = ['r', 'g', 'purple', 'orange', 'cyan']
        for i, peak in enumerate(peak_info):
            # Adjust the peak range by adding the shift value
            adjusted_range = (peak['range'][0] + shift_value, peak['range'][1] + shift_value)

            # Filter data for the current peak range
            df_range = df[(df['X'] >= adjusted_range[0]) & (df['X'] <= adjusted_range[1])]

            # Fit the Gaussian derivative to the data in the specified range
            if not df_range.empty:
                try:
                    # Initial guesses for fitting parameters
                    initial_A = 500
                    initial_mu = peak['initial_mu'] + shift_value
                    initial_sigma = 50
                    initial_C = df_range['Y'].mean()

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

                    # Extract fitted parameters and errors
                    A, mu, sigma, C = popt
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

                except RuntimeError as e:
                    print(f"Error fitting Gaussian derivative for peak {peak['name']}: {e}")

        plt.legend(fontsize=12)

        # Save the plot as a PNG file
        output_file_path = file_path.replace('.txt', '_Al_KLL_peaks_fit_with_baseline_plot.png')
        plt.savefig(output_file_path, format='png', dpi=300)
        plt.show()
        print(f"Plot with Gaussian fits saved as PNG: {output_file_path}")

        plt.close()  # Close the plot to free up memory

    else:
        print(f"No valid data found in {file_path}.")

# Example usage
file_path = './Auger_4keV_average5.txt'  # Replace with your actual file path
fit_al_kll_peaks_with_baseline(file_path)
