
import os
import numpy as np
import matplotlib.pyplot as plt

# Folder containing .dat files
folder_path = "."

# Iterate through all .dat files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".dat"):
        file_path = os.path.join(folder_path, filename)

        # Load the data (assume tab-separated and in scientific notation)
        data = np.loadtxt(file_path)

        # Extract columns
        x = data[:, 0]  # First column as x-axis
        y = data[:, 1]  # Second column as y-axis

        # Plot the data
        plt.plot(x, y, label=filename)

# Customize the plot
plt.xlabel("X-axis Label")
plt.ylabel("Y-axis Label")
plt.title("Plots from .dat files")
plt.legend()  # Show the legend
plt.grid(True)
plt.show()
