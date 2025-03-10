
import numpy as np
import matplotlib.pyplot as plt

# Data
power_first = np.array([24.88, 23.21, 21.3, 19.6, 17.92, 16.08, 14.09, 12.4, 10.84, 8.89,
                         6.93, 5.3, 3.76, 5.1, 1.428, 0.0974, 0.00123])

# Parameters
T = 0.02  # Transmission coefficient
L_values = np.arange(0, 0.11, 0.02)  # L values from 0% to 10% in steps of 2%

# Calculate Pcav for each L
Pcav_values = []
for L in L_values:
    Pcav = power_first / (T * (1 - L))
    Pcav_values.append(Pcav)

# Plotting
plt.figure(figsize=(10, 6))

for i, L in enumerate(L_values):
    plt.scatter(power_first, Pcav_values[i], label=f'$L = {L*100:.0f}\\%$')  # Use scatter plot for points

plt.title(r"$P_{\text{cav}}$ vs $P_{\text{out}}$ for different values of Loss ($L$)", fontsize=14)
plt.xlabel(r"$P_{\text{out}}$ (Nd:YAG Laser Power) [mW]", fontsize=12)
plt.ylabel(r"$P_{\text{cav}}$ [mW]", fontsize=12)
plt.grid(True)
plt.legend(title="Loss ($L$)", fontsize=12)
plt.savefig("Pcav_vs_Pout_points.pdf")
plt.show()
