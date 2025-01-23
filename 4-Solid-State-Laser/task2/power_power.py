
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Data
power_first = np.array([336, 315.8, 292.4, 270.5, 250.1, 227.4, 202.5, 182.2, 159.7, 131.8,
                        106.7, 85.44, 64.3, 82.46, 33.5, 10.48, 6.63])
power_second = np.array([24.88, 23.21, 21.3, 19.6, 17.92, 16.08, 14.09, 12.4, 10.84, 8.89,
                         6.93, 5.3, 3.76, 5.1, 1.428, 0.0974, 0.00123])

# Uncertainties
power_first_error = 4.82e-1
power_second_error = 2.21e-2

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(power_first, power_second)

# Generate fit line
fit_line = slope * power_first + intercept

# Propagate uncertainties
n = len(power_first)
weights = 1 / (power_second_error ** 2)  # Weights assuming constant uncertainty in power_second
X = np.vstack([np.ones(n), power_first]).T
cov_matrix = np.linalg.inv(X.T @ (weights * np.eye(n)) @ X)

# Extract standard errors for slope and intercept
slope_error = np.sqrt(cov_matrix[1, 1])
intercept_error = np.sqrt(cov_matrix[0, 0])

# Calculate chi-squared
residuals = power_second - fit_line
chi_squared = np.sum((residuals / power_second_error) ** 2)
reduced_chi_squared = chi_squared / (n - 2)  # Degrees of freedom = n - 2

# Plot data with error bars and fit line
plt.figure(figsize=(10, 6))
plt.errorbar(power_first, power_second, xerr=power_first_error, yerr=power_second_error,
             fmt='o', label='Data', color='b')
plt.plot(power_first, fit_line, label=(
    f'Linear fit: y = ({slope:.2f} ± {slope_error:.2e})x + ({intercept:.2f} ± {intercept_error:.2e})'
), color='r')
plt.title("Power-Power Plot with Linear Fit")
plt.xlabel("Diode Laser Power [mW]")
plt.ylabel("Nd:YAG Laser Power [mW]")
plt.grid(True)
plt.legend()
plt.savefig("power_power_plot.pdf")
plt.show()

# Print results
print(f"Slope: {slope:.4f} ± {slope_error:.4f}")
print(f"Intercept: {intercept:.4f} ± {intercept_error:.4f}")
print(f"Reduced chi-squared: {reduced_chi_squared:.4f}")
