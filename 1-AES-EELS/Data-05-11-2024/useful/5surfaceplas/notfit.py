
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Updated peak information with the new manually indicated mu values
peak_info = [
    {"name": "Excitation", "mu": 180},
    {"name": "S1", "mu": 170},
    {"name": "B1", "mu": 165},
    {"name": "S2", "mu": 158},
    {"name": "B2", "mu": 149},
    {"name": "S3", "mu": 141.5},
    {"name": "B3", "mu": 134},
    {"name": "S4", "mu": 124},
    {"name": "B4", "mu": 114}
]

# Shift value to add to all energy values (0, as per the request)
shift_value = 0

def plot_plasmons(file_path):
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
        plt.title("Data for Al Surface and Bulk Plasmons")
        plt.xlabel("Corrected Energy (eV)")
        plt.ylabel("dN/dE (a.u.)")
        #plt.grid()

        # List of colors for the vertical lines
        colors = ['r', 'g', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'blue']

        mu_values = []  # Store the mu values for distance calculation

        # Plot vertical lines for each mu value
        for i, peak in enumerate(peak_info):
            mu = peak['mu'] + shift_value  # Add shift to each mu value
            mu_values.append(mu)  # Store mu for distance calculation

            # Plot vertical line at `mu`
            plt.axvline(mu, color=colors[i % len(colors)], linestyle='--', label=f"{peak['name']} ({mu:.2f} eV)")

        # Calculate and print distances between consecutive mu values
        print("\nDistances between consecutive peaks:")
        for j in range(1, len(mu_values)):
            distance = abs(mu_values[j] - mu_values[j - 1])
            print(f"Distance between {peak_info[j-1]['name']} and {peak_info[j]['name']}: {distance:.2f} eV")

            # Draw arrows between consecutive mu values
            plt.annotate('', xy=(mu_values[j], max(df['Y'])), xytext=(mu_values[j - 1], max(df['Y'])),
                         arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))

            # Annotate the distance along the arrow
            mid_point = (mu_values[j] + mu_values[j - 1]) / 2
            vertical_offset = df['Y'].max() * 0.2 +8000  # Adjust the offset for annotation
            plt.text(mid_point, vertical_offset, f"{distance:.0f} eV", color='black', ha='center', va='bottom')

        plt.legend()  # Ensure the legend is added after the lines are plotted

        # Save the plot as a PDF file (if needed)
        output_file_path = file_path.replace('.txt', '_plasmons_plot.pdf')
        plt.savefig(output_file_path, format='pdf')
        plt.show()
        print(f"Plot saved as PDF: {output_file_path}")

        plt.close()  # Close the plot to free up memory
    else:
        print(f"No valid data found in {file_path}.")

# Example usage:
file_path = './surfaceplasmons_30uV_156deg_300ms_1.9e-09mbar_200eV.txt'  # Replace with your actual file path
plot_plasmons(file_path)
