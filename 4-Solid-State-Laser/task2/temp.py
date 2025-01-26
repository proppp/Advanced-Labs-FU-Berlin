
import numpy as np
import matplotlib.pyplot as plt

# Data
temperature = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                        30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40])
power = np.array([11.96, 11.23, 11.36, 11.17, 11.35, 11.28, 14.06, 16.56, 15.99, 19.22,
                  23.01, 23.2, 23.99, 25.14, 25.13, 24.53, 24.45, 24.31, 24.57, 24.54,
                  24.42, 24.6, 24.57, 24.69, 23.85, 23.82, 23.99, 20.29, 20.85, 20.19, 17.3])

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(temperature, power, color='b', label='Data', marker='o')

# Adjust grid spacing
plt.title('Temperature vs Power')
plt.xlabel('Temperature (°C)')
plt.ylabel('Power (mW)')

# Increase grid spacing by setting major and minor ticks
plt.xticks(np.arange(min(temperature), max(temperature)+1, 5))  # Every 2°C on x-axis
plt.yticks(np.arange(min(power), max(power)+1, 5))  # Every 2mW on y-axis

plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)

plt.legend()
plt.savefig("temperature_vs_power_scatter_larger_grid.pdf")
plt.show()
