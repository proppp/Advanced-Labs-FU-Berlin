
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the specified x-values and corresponding labels
x_values = [200, 192, 183, 167, 154, 143]
labels = ["Excitation", "S1", "B1", "B2", "B3", "B4"]

# Shift value to add to all energy values (15.74 eV)
shift_value = 0

def plot_vertical_lines_with_distances(file_path):
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
        plt.title("Data with Marked X-values and Distances")
        plt.xlabel("Corrected Energy (eV)")
        plt.ylabel("dN/dE (a.u.)")

        # Plot vertical lines and add labels
        for i, (x_val, label) in enumerate(zip(x_values, labels)):
            plt.axvline(x_val, color='r', linestyle='--', alpha=0.7)
            plt.text(x_val, df['Y'].max() * 0.9, label, rotation=90, color='r', verticalalignment='bottom')

        # Calculate and display distances between consecutive x-values
        print("\nDistances between consecutive peaks:")
        for j in range(1, len(x_values)):
            distance = abs(x_values[j] - x_values[j - 1])
            print(f"Distance between {labels[j - 1]} and {labels[j]}: {distance:.2f} eV")

            # Draw arrows between consecutive x-values
            plt.annotate('', xy=(x_values[j], df['Y'].max() * 0.8),
                         xytext=(x_values[j - 1], df['Y'].max() * 0.8),
                         arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))

            # Annotate the distance along the arrow
            mid_point = (x_values[j] + x_values[j - 1]) / 2
            vertical_offset = df['Y'].max() * (0.75 - 0.05 * (j % 2))  # Alternating offsets for clarity
            plt.text(mid_point, vertical_offset, f"{distance:.2f} eV", color='black', ha='center', va='bottom')

        plt.legend(["Data"])
        plt.grid()

        # Save the plot as a PDF file
        output_file_path = file_path.replace('.txt', '_Al_KLL_peaks_marked_distances.pdf')
        plt.savefig(output_file_path, format='pdf')
        plt.show()
        print(f"Plot saved as PDF: {output_file_path}")

        plt.close()  # Close the plot to free up memory

    else:
        print(f"No valid data found in {file_path}.")

# Example usage:
file_path = './surface3.txt'  # Replace with your actual file path
plot_vertical_lines_with_distances(file_path)
