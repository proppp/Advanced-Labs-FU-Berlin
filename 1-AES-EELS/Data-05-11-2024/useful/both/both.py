
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Define the derivative of the Gaussian function
def gaussian_derivative(x, A, mu, sigma):
    return A * (-2 * (x - mu) / sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def plot_two_spectra_with_selected_dashed_lines(folder_path, ranges, offset=25000):
    # Define custom labels for the legend based on ranges
    labels = [
        "Al L₂,₃VV (μ = {:.2f} eV)",
        "C KVV (μ = {:.2f} eV)",
        "O KVV (μ = {:.2f} eV)",
        "Al KL₂,₃L₂,₃ (μ = {:.2f} eV)"
    ]

    # Identify cleaned and uncleaned sample files
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    cleaned_file = next((f for f in txt_files if 'cleaned' in f.lower()), None)
    uncleaned_file = next((f for f in txt_files if 'dirty' in f.lower()), None)

    if not cleaned_file or not uncleaned_file:
        print("Both 'cleaned' and 'uncleaned' sample files must be present in the folder.")
        return

    # Helper function to read and process data
    def read_data(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        data = []
        for line in lines[6:]:  # Start from the 7th line onward
            if line.strip():
                parts = line.split()
                if len(parts) == 2:
                    try:
                        x_value = float(parts[0])
                        y_value = float(parts[1])
                        data.append((x_value, y_value))
                    except ValueError:
                        print(f"Skipping line due to conversion error: {line}")
        return pd.DataFrame(data, columns=['X', 'Y']) if data else None

    # Read data for both samples
    cleaned_data = read_data(os.path.join(folder_path, cleaned_file))
    uncleaned_data = read_data(os.path.join(folder_path, uncleaned_file))

    if cleaned_data is None or uncleaned_data is None:
        print("No valid data found in one or both files.")
        return

    # Plot the spectra for both samples
    plt.figure(figsize=(10, 6))

    # Plot cleaned data
    plt.plot(cleaned_data['X'], cleaned_data['Y'], linestyle='-', color='b', label='Cleaned Sample Data')

    # Plot uncleaned data with an offset for visibility
    plt.plot(uncleaned_data['X'], uncleaned_data['Y'] + offset, linestyle='-', color='orange', label='Uncleaned Sample Data')

    # Fit Gaussian derivatives and plot vertical dashed lines at each μ
    colors = ['r', 'g', 'purple', 'orange']  # Colors for each range
    for i, x_range in enumerate(ranges):
        # Variable to store only the "dirty" mu for C KVV and O KVV transitions
        use_dirty_mu_only = (i == 1 or i == 2)  # Carbon (index 1) and Oxygen (index 2)

        # Initialize mu to None for both samples
        mu_dirty = None
        mu_clean = None

        for sample_data, sample_label in zip([cleaned_data, uncleaned_data], ["Cleaned", "Uncleaned"]):
            # Filter data for the current range
            df_range = sample_data[(sample_data['X'] >= x_range[0]) & (sample_data['X'] <= x_range[1])]

            # Fit the Gaussian derivative to the data in the specified range
            if not df_range.empty:
                try:
                    popt, _ = curve_fit(
                        gaussian_derivative,
                        df_range['X'],
                        df_range['Y'],
                        p0=[1, np.mean(df_range['X']), np.std(df_range['X'])]
                    )
                    _, mu, _ = popt
                    print(f"{sample_label} Sample - Fitted μ for Range {i+1} ({x_range}): μ = {mu}")

                    # Store mu value for dirty (uncleaned) and clean samples
                    if sample_label == "Uncleaned":
                        mu_dirty = mu
                    elif sample_label == "Cleaned":
                        mu_clean = mu

                except RuntimeError as e:
                    print(f"Error fitting Gaussian derivative for range {i+1} in {sample_label} Sample: {e}")

        # Plot the vertical dashed lines at μ values as specified
        color = colors[i % len(colors)]
        if use_dirty_mu_only and mu_dirty is not None:
            # Plot only the "dirty" sample's mu for Carbon and Oxygen
            plt.axvline(mu_dirty, color=color, linestyle='--', linewidth=1.5,
                        label=labels[i].format(mu_dirty))
        elif not use_dirty_mu_only:
            # Plot both "dirty" and "clean" mu values for Aluminum transitions
            if mu_dirty is not None:
                plt.axvline(mu_dirty, color=color, linestyle='--', linewidth=1.5,
                            label=f"Uncleaned {labels[i].format(mu_dirty)}")
            if mu_clean is not None:
                plt.axvline(mu_clean, color=color, linestyle='-.', linewidth=1.5,
                            label=f"Cleaned {labels[i].format(mu_clean)}")

    # Set plot title and labels
    plt.title("Cleaned and Uncleaned Samples with Gaussian Derivative Peak Positions")
    plt.xlabel("Uncorrected Energy (eV)")
    plt.ylabel("dN/dE (a.u.)")
    plt.legend()
    #plt.grid()

    # Save the plot as a PDF file
    output_file_path = os.path.join(folder_path, "cleaned_uncleaned_peak_positions_selected_plot.pdf")
    plt.savefig(output_file_path, format='pdf')
    print(f"Plot saved as PDF: {output_file_path}")

    plt.show()
    plt.close()  # Close the plot to free up memory

# Example usage:
folder_path = '.'  # Replace with your actual folder path
ranges = [(10, 105), (205, 320), (492, 542), (1364, 1385)]  # Specify the four ranges here
plot_two_spectra_with_selected_dashed_lines(folder_path, ranges)
