
import numpy as np
import matplotlib.pyplot as plt

# Data (X and Y inverted)
power_second = np.array([336, 315.8, 292.4, 270.5, 250.1, 227.4, 202.5, 182.2, 159.7, 131.8,
                        106.7, 85.44, 64.3, 82.46, 33.5, 10.48, 6.63])
power_532_1 = np.array([17.2, 15.3, 13.15, 11.22, 10.38, 9.41, 7.4, 6.39, 5.08, 3.65,
                        2.42, 1.454, 0.748, 1.299, 0.0766, 0.000185, 0.0001839])  # Initial position of KTP
power_532_2 = np.array([12.9, 11.5, 10.14, 8.7, 7.23, 6.2, 4.95, 4.21, 3.25, 2.22,
                        1.3, 0.783, 0.41, 0.7, 0.05, 0.000185, 0.00017])  # Moved KTP crystal away from Nd:YAG

power_1064 = np.array([24.88, 23.21, 21.3, 19.6, 17.92, 16.08, 14.09, 12.4, 10.84, 8.89,
                       6.93, 5.3, 3.76, 5.1, 1.428, 0.0974, 0.00123])

# Parameters
T = 1.0  # Transmission coefficient set to 100% (1.0)
L = 0.05  # Loss L = 5%

# Compute efficiencies for L = 5% and T = 100%
Pcav_1064 = power_1064 / (T * (1 - L))
Pcav_532_1 = power_532_1 / (T * (1 - L))
Pcav_532_2 = power_532_2 / (T * (1 - L))

efficiency_1 = Pcav_532_1 / Pcav_1064
efficiency_2 = Pcav_532_2 / Pcav_1064

# Define uncertainties (5%)
uncertainty_percentage = 0.005
power_second_error = power_second * uncertainty_percentage
power_532_1_error = power_532_1 * uncertainty_percentage
power_532_2_error = power_532_2 * uncertainty_percentage

# Plot efficiencies with error bars (crosses)
plt.figure(figsize=(10, 6))
plt.errorbar(power_second, efficiency_1, xerr=power_second_error, yerr=power_532_1_error,
             fmt='o', linestyle='-', label=f'Initial KTP, L={L*100:.0f}%', capsize=5)
plt.errorbar(power_second, efficiency_2, xerr=power_second_error, yerr=power_532_2_error,
             fmt='s', linestyle='--', label=f'Moved KTP, L={L*100:.0f}%', capsize=5)

plt.xlabel("Intracavity Power (mW)")
plt.ylabel("SHG Efficiency")
plt.title(f"Efficiency for L={L*100:.0f}% and T=100% with 5% Uncertainty")
plt.legend()
plt.grid(True)
plt.show()
