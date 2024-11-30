
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Define the derivative of the Gaussian function
def gaussian_derivative(x, A, mu, sigma):
    return A * (-2 * (x - mu) / sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def plot_txt_files_in_folder(folder_path, x_range=None):
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

            # If a specific x range is provided, filter the data
            if x_range:
                df = df[(df['X'] >= x_range[0]) & (df['X'] <= x_range[1])]

            # Fit the derivative of a Gaussian function
            try:
                # Fit the Gaussian derivative to the data
                popt, pcov = curve_fit(gaussian_derivative, df['X'], df['Y'], p0=[1, np.mean(df['X']), np.std(df['X'])])
                A, mu, sigma = popt
                print(f"Fitted Gaussian Derivative Parameters for {file_name}: A = {A}, mu = {mu}, sigma = {sigma}")

                # Generate the fit curve
                fit_curve = gaussian_derivative(df['X'], *popt)

                # Plot the data and the fitted curve
                plt.figure(figsize=(10, 6))
                plt.plot(df['X'], df['Y'], marker='o', linestyle='-', color='b', label='Data')
                plt.plot(df['X'], fit_curve, color='r', linestyle='--', label='Gaussian Derivative Fit')
                plt.title(f'Data and Gaussian Derivative Fit for {file_name}')
                plt.xlabel('X Values')
                plt.ylabel('Y Values')
                plt.grid()
                plt.legend()

                # Set x-axis limits based on the data or user-specified range
                if x_range:
                    plt.xlim(x_range[0], x_range[1])

                # Show the plot for the current file
                plt.show()

            except RuntimeError as e:
                print(f"Error fitting Gaussian derivative for {file_name}: {e}")
        else:
            print(f"No valid data found in {file_name}.")

# Example usage:
folder_path = '.'  # Replace with your actual folder path
x_range = (196, 209)  # Specify the range for the x-axis (min, max)
plot_txt_files_in_folder(folder_path, x_range)
