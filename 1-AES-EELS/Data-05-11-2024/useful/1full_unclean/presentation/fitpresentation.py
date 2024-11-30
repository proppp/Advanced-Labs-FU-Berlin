
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Define the derivative of the Gaussian function
def gaussian_derivative(x, A, mu, sigma):
    return A * (-2 * (x - mu) / sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def plot_txt_files_with_multiple_fits(folder_path, ranges, energy_correction=15.74):
    # Apply 15.4 eV correction to each range
    adjusted_ranges = [(start + 15.4, end + 15.4) for start, end in ranges]

    # Define custom labels for the legend based on ranges
    labels = [
        "Gaussian derivative Al L₂,₃VV ({:.2f} eV)",
        "Gaussian derivative C KVV ({:.2f} eV)",
        "Gaussian derivative O KVV ({:.2f} eV)",
        "Gaussian derivative Al KL₂,₃L₂,₃ ({:.2f} eV)"
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

        # Read the data section and apply the energy correction
        data = []
        for line in lines[6:]:  # Start from the 7th line onward
            if line.strip():  # Ensure line is not empty
                parts = line.split()
                if len(parts) == 2:  # Expect two values per line
                    try:
                        x_value = float(parts[0]) + energy_correction  # Apply correction
                        y_value = float(parts[1])
                        data.append((x_value, y_value))
                    except ValueError:
                        print(f"Skipping line due to conversion error: {line}")

        # Convert data to DataFrame for easier plotting
        if data:
            df = pd.DataFrame(data, columns=['X', 'Y'])

            # Plot the original data as a continuous line (without Gaussian fits)
            plt.figure(figsize=(10, 6))
            plt.plot(df['X'], df['Y'], linestyle='-', color='b', label='Data')
            plt.title("Energy Spectrum (Uncleaned Sample)", fontsize=16)
            plt.xlabel("Energy (eV)", fontsize=14)
            plt.ylabel("dN/dE (a.u.)", fontsize=14)
            #plt.grid()
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(fontsize=12)

            # Save the plot without Gaussian fits as PNG
            output_file_path_no_fit = os.path.join(folder_path, f"{file_name}_plot.png")
            plt.savefig(output_file_path_no_fit, format='png', dpi=300)
            print(f"Plot saved without Gaussian fits: {output_file_path_no_fit}")
            plt.close()

            # Plot the original data with Gaussian fits
            plt.figure(figsize=(10, 6))
            plt.plot(df['X'], df['Y'], linestyle='-', color='b', label='Data')
            plt.title("Energy Spectrum with Gaussian Derivative Fits", fontsize=16)
            plt.xlabel("Energy (eV)", fontsize=14)
            plt.ylabel("dN/dE (a.u.)", fontsize=14)
            #plt.grid()
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            # Colors for Gaussian derivative fits
            colors = ['r', 'g', 'purple', 'orange']
            for i, x_range in enumerate(adjusted_ranges):
                # Filter data for the current range
                df_range = df[(df['X'] >= x_range[0]) & (df['X'] <= x_range[1])]

                # Fit the Gaussian derivative to the data in the specified range
                if not df_range.empty:
                    try:
                        # Fit the Gaussian derivative function
                        popt, _ = curve_fit(
                            gaussian_derivative,
                            df_range['X'],
                            df_range['Y'],
                            p0=[1, np.mean(df_range['X']), np.std(df_range['X'])]
                        )
                        A, mu, sigma = popt
                        print(f"Fitted Parameters for Range {i+1} ({x_range}): A = {A}, mu = {mu}, sigma = {sigma}")

                        # Generate the fit curve
                        fit_curve = gaussian_derivative(df_range['X'], *popt)

                        # Plot the fit curve with a unique color
                        plt.plot(df_range['X'], fit_curve, linestyle='-', color=colors[i % len(colors)],
                                 label=labels[i].format(mu))

                    except RuntimeError as e:
                        print(f"Error fitting Gaussian derivative for range {i+1} in {file_name}: {e}")

            # Add legend with larger font
            plt.legend(fontsize=12)

            # Save the plot with Gaussian fits as PNG
            output_file_path_with_fit = os.path.join(folder_path, f"{file_name}_fit_plot.png")
            plt.savefig(output_file_path_with_fit, format='png', dpi=300)
            print(f"Plot saved with Gaussian fits: {output_file_path_with_fit}")
            plt.close()  # Close the plot to free up memory

        else:
            print(f"No valid data found in {file_name}.")

# Example usage:
folder_path = '.'  # Replace with your actual folder path
#ranges = [(10, 105), (205, 320), (492, 542), (1364, 1385)]  # Specify the four ranges here

ranges = [(58, 105), (266, 310), (492, 542), (1368, 1385)]  # Specify the four ranges here
plot_txt_files_with_multiple_fits(folder_path, ranges)
