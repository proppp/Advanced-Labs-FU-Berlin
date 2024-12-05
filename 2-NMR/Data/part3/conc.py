
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Data for concentration (M) and relaxation times (T1 and T2) for CuSO4
concentrations = np.array([0.05, 0.1, 0.2])

# T1 values (in seconds) for IR method at each concentration
T1_values = np.array([0.022, 0.011, 0.005])

# T2 values (in seconds) for SE, CP, and MG methods at each concentration
T2_SE = np.array([0.0199, 0.0101, 0.0045])
T2_CP = np.array([0.0024, 0.0038, 0.0024])
T2_MG = np.array([0.0186, 0.0102, 0.0042])

# Plotting
plt.figure(figsize=(10, 6))

# Plot T1 vs concentration
plt.subplot(2, 1, 1)
plt.plot(concentrations, T1_values, 'bo-', label="T1 (IR)", markersize=8)
plt.xlabel("Concentration (M)")
plt.ylabel("Relaxation Time T1 (s)")
plt.title("Relaxation Time T1 vs. Concentration of CuSO4")
plt.grid(True)

# Plot T2 vs concentration for SE, CP, and MG
plt.subplot(2, 1, 2)
plt.plot(concentrations, T2_SE, 'ro-', label="T2 (SE)", markersize=8)
plt.plot(concentrations, T2_CP, 'go-', label="T2 (CP)", markersize=8)
plt.plot(concentrations, T2_MG, 'yo-', label="T2 (MG)", markersize=8)
plt.xlabel("Concentration (M)")
plt.ylabel("Relaxation Time T2 (s)")
plt.title("Relaxation Time T2 vs. Concentration of CuSO4")
plt.grid(True)

# Linear fit for T1
slope_T1, intercept_T1, r_value_T1, p_value_T1, std_err_T1 = linregress(concentrations, T1_values)
plt.subplot(2, 1, 1)
plt.plot(concentrations, slope_T1 * concentrations + intercept_T1, 'b--', label=f"Fit: T1 = {slope_T1:.4f}x + {intercept_T1:.4f}")

# Linear fit for T2 (SE)
slope_T2_SE, intercept_T2_SE, r_value_T2_SE, p_value_T2_SE, std_err_T2_SE = linregress(concentrations, T2_SE)
plt.subplot(2, 1, 2)
plt.plot(concentrations, slope_T2_SE * concentrations + intercept_T2_SE, 'r--', label=f"Fit: T2 (SE) = {slope_T2_SE:.4f}x + {intercept_T2_SE:.4f}")

# Linear fit for T2 (CP)
slope_T2_CP, intercept_T2_CP, r_value_T2_CP, p_value_T2_CP, std_err_T2_CP = linregress(concentrations, T2_CP)
plt.subplot(2, 1, 2)
plt.plot(concentrations, slope_T2_CP * concentrations + intercept_T2_CP, 'g--', label=f"Fit: T2 (CP) = {slope_T2_CP:.4f}x + {intercept_T2_CP:.4f}")

# Linear fit for T2 (MG)
slope_T2_MG, intercept_T2_MG, r_value_T2_MG, p_value_T2_MG, std_err_T2_MG = linregress(concentrations, T2_MG)
plt.subplot(2, 1, 2)
plt.plot(concentrations, slope_T2_MG * concentrations + intercept_T2_MG, 'y--', label=f"Fit: T2 (MG) = {slope_T2_MG:.4f}x + {intercept_T2_MG:.4f}")

# Displaying the legend
plt.subplot(2, 1, 1)
plt.legend(loc="best")
plt.subplot(2, 1, 2)
plt.legend(loc="best")

plt.tight_layout()
plt.savefig("relaxation_times_vs_concentration.pdf", format="pdf")

# Return the proportionality factors (slopes)
(slope_T1, slope_T2_SE, slope_T2_CP, slope_T2_MG)
