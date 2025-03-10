
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Data for concentrations and relaxation times (T1 and T2)
concentrations = np.array([0.05, 0.1, 0.2])  # in M

# IR: T1 (s)
T1_IR = np.array([0.022, 0.011, 0.005])

# SE: T2 (s)
T2_SE = np.array([0.0165, 0.0103, 0.0054])

# CP: T2 (s)
T2_CP = np.array([0.0029, 0.0021, 0.0023])

# MG: T2 (s)
T2_MG = np.array([0.0193, 0.0102, 0.0046])

# CuSO4: T2 (s)
T2_CuSO4 = np.array([0.0165, 0.0103, 0.0054])

# Calculate 1/T for each method
inv_T1_IR = 1 / T1_IR
inv_T2_SE = 1 / T2_SE
inv_T2_CP = 1 / T2_CP
inv_T2_MG = 1 / T2_MG
inv_T2_CuSO4 = 1 / T2_CuSO4

# Fit linear models to determine proportionality factors (for 1/T1 and 1/T2)
slope_inv_T1, intercept_inv_T1, _, _, _ = stats.linregress(concentrations, inv_T1_IR)
slope_inv_T2_SE, intercept_inv_T2_SE, _, _, _ = stats.linregress(concentrations, inv_T2_SE)
slope_inv_T2_CP, intercept_inv_T2_CP, _, _, _ = stats.linregress(concentrations, inv_T2_CP)
slope_inv_T2_MG, intercept_inv_T2_MG, _, _, _ = stats.linregress(concentrations, inv_T2_MG)
slope_inv_T2_CuSO4, intercept_inv_T2_CuSO4, _, _, _ = stats.linregress(concentrations, inv_T2_CuSO4)

# Plotting
plt.figure(figsize=(10, 6))

# Plot 1/T1 vs Concentration for IR
plt.plot(concentrations, inv_T1_IR, 'o-', label=f'IR $1/T_1$: {slope_inv_T1:.4f} (1/s/M)', color='blue')

# Plot 1/T2 vs Concentration for SE, CP, MG, CuSO4
plt.plot(concentrations, inv_T2_SE, 's-', label=f'SE $1/T_2$: {slope_inv_T2_SE:.4f} (1/s/M)', color='green')
plt.plot(concentrations, inv_T2_CP, '^-', label=f'CP $1/T_2$: {slope_inv_T2_CP:.4f} (1/s/M)', color='red')
plt.plot(concentrations, inv_T2_MG, 'D-', label=f'MG $1/T_2$: {slope_inv_T2_MG:.4f} (1/s/M)', color='purple')
plt.plot(concentrations, inv_T2_CuSO4, 'x-', label=f'CuSO$_4$ $1/T_2$: {slope_inv_T2_CuSO4:.4f} (1/s/M)', color='orange')

# Labeling the plot
plt.xlabel('Concentration (M)')
plt.ylabel('Inverse Relaxation Time (1/s)')
plt.title('Inverse Relaxation Times vs Concentration for CuSO$_4$')
plt.legend()

# Show the plot
plt.show()
