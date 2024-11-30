
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
    {"name": "K-L$_{2,3}$L$_{2,3}$ ($^{1}$D)", "initial_mu": 1393.3, "range": (1372, 1385)},
    {"name": "K-L$_{2,3}$L$_{2,3}$ ($^{1}$D) + B1", "initial_mu": 1386.85, "range": (1354, 1370)},
    {"name": "K-L$_{1}$L$_{2,3}$ ($^{3}$P)", "initial_mu": 1357.85, "range": (1338, 1351)},
    {"name": "K-L$_{1}$L$_{2,3}$ ($^{1}$P)", "initial_mu": 1341.95, "range": (1320, 1336)},
    {"name": "K-L$_{1}$L$_{2,3}$ ($^{1}$P) + B1", "initial_mu": 1303.3, "range": (1296, 1318)}
]

# Define the assumed error in X (e.g., 0.5 eV)
error_x = 1.5

def fit_al_kll_peaks_with_baseline(file_path, fixed_peak_value=1393.5):
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
                    x_value = float(parts[0])  # Initially unshifted energy
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
        plt.title("Data and Gaussian Derivative Fits for Al KLL Transitions")
        plt.xlabel("Uncorrected Energy (eV)")
        plt.ylabel("dN/dE (a.u.)")
        #plt.grid()

        fitted_mus = []  # Store fitted mu values for energy correction
        colors = ['r', 'g', 'purple', 'orange', 'cyan']  # Colors for each peak

        for i, peak in enumerate(peak_info):
            # Filter data for the current peak range
            df_range = df[(df['X'] >= peak['range'][0]) & (df['X'] <= peak['range'][1])]

            # Fit the Gaussian derivative to the data in the specified range
            if not df_range.empty:
                try:
                    # Set initial guesses for fitting parameters
                    initial_A = 500  # Maximal expected height
                    initial_mu = peak['initial_mu']  # Use initial_mu directly
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

                    # Calculate the errors (standard deviation) of the fitted parameters
                    perr = np.sqrt(np.diag(pcov))

                    print(f"Fitted Parameters for {peak['name']}:")
                    print(f"A = {A:.2f} ± {perr[0]:.2f}")
                    print(f"mu = {mu:.2f} ± {perr[1]:.2f}")
                    print(f"sigma = {sigma:.2f} ± {perr[2]:.2f}")
                    print(f"C = {C:.2f} ± {perr[3]:.2f}")

                    # Store the fitted `mu` value
                    fitted_mus.append((mu, perr[1]))

                    # Generate the fit curve
                    fit_curve = gaussian_derivative_with_baseline(df_range['X'], *popt)

                    # Plot the fit curve with a unique color and label with fitted mu
                    plt.plot(df_range['X'], fit_curve, linestyle='-', color=colors[i % len(colors)],
                             label=f"{peak['name']} ({mu:.2f} eV)")

                except RuntimeError as e:
                    print(f"Error fitting Gaussian derivative for peak {peak['name']}: {e}")

        # Apply the energy correction based on the last peak's fitted mu
        if fitted_mus:
            last_mu, _ = fitted_mus[0]  # Assume the first peak corresponds to the fixed reference
            shift = fixed_peak_value - last_mu
            corrected_mus = [(mu + shift, error) for mu, error in fitted_mus]

            print("\nCorrected Energy Values:")
            for i, (corrected_mu, error) in enumerate(corrected_mus):
                print(f"Peak {i+1} ({peak_info[i]['name']}): Corrected Energy = {corrected_mu:.2f} ± {error:.2f} eV")

        plt.legend()

        # Save the plot as a PDF file
        output_file_path = file_path.replace('.txt', '_Al_KLL_peaks_fit_with_baseline_plot_corrected.pdf')
        plt.savefig(output_file_path, format='pdf')
        print(f"Plot saved as PDF: {output_file_path}")

        plt.close()  # Close the plot to free up memory

    else:
        print(f"No valid data found in {file_path}.")

# Example usage:
file_path = './Auger_4keV_average5.txt'  # Replace with your actual file path
fit_al_kll_peaks_with_baseline(file_path)
