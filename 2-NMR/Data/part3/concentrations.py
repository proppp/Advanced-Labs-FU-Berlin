
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Data for concentrations and relaxation times (T1 and T2)
concentrations = np.array([0.05, 0.1, 0.2])  # in M

# IR: T1 (s)
T1_IR = np.array([0.022, 0.011, 0.005])

# SE: T2 (s)
T2_SE = np.array([0.0199, 0.0101, 0.0045])

# CP: T2 (s)
T2_CP = np.array([0.0024, 0.0038, 0.0024])

# MG: T2 (s)
T2_MG = np.array([0.0186, 0.0102, 0.0042])

# CuSO4: T2 (s)
T2_CuSO4 = np.array([0.0199, 0.0101, 0.0045])

# Fit linear models to determine proportionality factors (for T1 and T2)
slope_T1, intercept_T1, _, _, _ = stats.linregress(concentrations, T1_IR)
slope_T2_SE, intercept_T2_SE, _, _, _ = stats.linregress(concentrations, T2_SE)
slope_T2_CP, intercept_T2_CP, _, _, _ = stats.linregress(concentrations, T2_CP)
slope_T2_MG, intercept_T2_MG, _, _, _ = stats.linregress(concentrations, T2_MG)
slope_T2_CuSO4, intercept_T2_CuSO4, _, _, _ = stats.linregress(concentrations, T2_CuSO4)

# Plotting
plt.figure(figsize=(10, 6))

# Plot T1 vs Concentration for IR
plt.plot(concentrations, T1_IR, 'o-', label=f'IR $T_1$: {slope_T1:.4f} (s/M)', color='blue')

# Plot T2 vs Concentration for SE, CP, MG, CuSO4
plt.plot(concentrations, T2_SE, 's-', label=f'SE $T_2$: {slope_T2_SE:.4f} (s/M)', color='green')
plt.plot(concentrations, T2_CP, '^-', label=f'CP $T_2$: {slope_T2_CP:.4f} (s/M)', color='red')
plt.plot(concentrations, T2_MG, 'D-', label=f'MG $T_2$: {slope_T2_MG:.4f} (s/M)', color='purple')
plt.plot(concentrations, T2_CuSO4, 'x-', label=f'CuSO$_4$ $T_2$: {slope_T2_CuSO4:.4f} (s/M)', color='orange')

# Labeling the plot
plt.xlabel('Concentration (M)')
plt.ylabel('Relaxation Time (s)')
plt.title('Relaxation Times vs Concentration for CuSO$_4$')
plt.legend()

# Show the plot
plt.show()
