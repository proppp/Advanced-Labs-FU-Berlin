
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Define the derivative of the Gaussian function with a vertical offset
def gaussian_derivative(x, A, mu, sigma, C):
    return A * (-2 * (x - mu) / sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2)) + C

def plot_and_correct_energy(folder_path, ranges, fixed_peak_value=1393.5):
    # Define custom labels for the legend based on ranges
    labels = [
        "Al L₂,₃VV ({:.2f}eV)",
        "B KVV ({:.2f}eV)",  # Replacing "thsi" with "B KVV"
        "C KVV ({:.2f}eV)",
        "N KVV ({:.2f}eV)",  # Replacing "thsi2" with "N KVV"
        "O KVV ({:.2f}eV)",
        "Al KL₂,₃L₂,₃ ({:.2f}eV)"
    ]

    # Get a list of all .txt files in the specified folder
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    if not txt_files:
        print("No .txt files found in the specified folder.")
        return

    for file_name in txt_files:
        file_path = os.path.join(folder_path, file_name)

        # Read the file and skip the header
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Extract header information
        header_info = {}
        for line in lines[:6]:  # The first six lines contain header info
            if ':' in line:  # Check if the line has a key-value pair
                key, value = line.split(':', 1)
                header_info[key.strip()] = value.strip()

        # Print header information for reference (optional)
        print(f"\nHeader Information for {file_name}:")
        for key, value in header_info.items():
            print(f"{key}: {value}")

        # Read the data section
        data = []
        for line in lines[6:]:  # Start from the 7th line onward
            if line.strip():  # Ensure line is not empty
                parts = line.split()
                if len(parts) == 2:  # We expect two values per line
                    try:
                        x_value = float(parts[0])
                        y_value = float(parts[1])
                        data.append((x_value, y_value))
                    except ValueError:
                        print(f"Skipping line due to conversion error: {line}")

        # Convert the data to a DataFrame for easier plotting
        if data:
            df = pd.DataFrame(data, columns=['X', 'Y'])

            # Calculate the spacing in `X`
            x_spacing = np.mean(np.diff(df['X']))
            print(x_spacing)

            # Plot the original data as a continuous line
            plt.figure(figsize=(10, 6))
            plt.plot(df['X'], df['Y'], linestyle='-', color='b', label='Data')
            plt.title("Data and Gaussian Derivative Fits for the Uncleaned Sample")
            plt.xlabel("Uncorrected Energy (eV)")
            plt.ylabel("dN/dE (a.u.)")
            #plt.grid()

            fitted_mus = []  # To store fitted `mu` values
            colors = ['r', 'g', 'purple', 'orange']  # Colors for each fit
            for i, x_range in enumerate(ranges):
                # Filter data for the current range
                df_range = df[(df['X'] >= x_range[0]) & (df['X'] <= x_range[1])]

                # Fit the Gaussian derivative to the data in the specified range
                if not df_range.empty:
                    try:
                        # Fit the Gaussian derivative function with offset
                        popt, pcov = curve_fit(
                            gaussian_derivative,
                            df_range['X'],
                            df_range['Y'],
                            p0=[10000, np.mean(df_range['X']), np.std(df_range['X']), 0.0]
                        )
                        A, mu, sigma, C = popt

                        # Extract errors from the diagonal of the covariance matrix
                        errors = np.sqrt(np.diag(pcov))
                        mu_error = errors[1] if len(errors) > 1 else float('nan')

                        # Store the fitted `mu` value
                        fitted_mus.append((mu, mu_error))

                        print(f"Fitted Parameters for Range {i+1} ({x_range}):")
                        print(f"  A = {A}, mu = {mu:.2f} ± {mu_error:.2f}, sigma = {sigma}, C = {C}")

                        # Generate the fit curve
                        fit_curve = gaussian_derivative(df_range['X'], *popt)

                        # Plot the fit curve with a unique color
                        plt.plot(df_range['X'], fit_curve, linestyle='-', color=colors[i % len(colors)],
                                 label=labels[i].format(mu))

                    except RuntimeError as e:
                        print(f"Error fitting Gaussian derivative for range {i+1} in {file_name}: {e}")

            # Correct all `mu` values by aligning the last peak
            if fitted_mus:
                last_mu, _ = fitted_mus[-1]  # Use the last fitted `mu` value
                shift = fixed_peak_value - last_mu
                corrected_mus = [(mu + shift, error) for mu, error in fitted_mus]

                print("\nCorrected Energy Values:")
                for i, (corrected_mu, mu_error) in enumerate(corrected_mus):
                    print(f"Range {i+1}: Corrected Energy = {corrected_mu:.2f} ± {mu_error:.2f} eV")

            plt.legend()

            # Save the plot as a PDF file
            output_file_path = os.path.join(folder_path, f"{file_name}_fit_plot.pdf")
            plt.savefig(output_file_path, format='pdf')
            print(f"Plot saved as PDF: {output_file_path}")

            plt.close()  # Close the plot to free up memory

        else:
            print(f"No valid data found in {file_name}.")

# Example usage:
folder_path = '.'  # Replace with your actual folder path
ranges = [(58, 105), (143, 160), (266, 305),(367, 400), (492, 542), (1368, 1385)]  # Specify the four ranges here
plot_and_correct_energy(folder_path, ranges)
