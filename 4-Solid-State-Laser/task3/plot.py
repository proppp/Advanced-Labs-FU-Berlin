
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data (X and Y inverted)
power_second = np.array([24.88, 23.21, 21.3, 19.6, 17.92, 16.08, 14.09, 12.4, 10.84, 8.89,
                         6.93, 5.3, 3.76, 5.1, 1.428, 0.0974, 0.00123])
power_first_1 = np.array([17.2, 15.3, 13.15, 11.22, 10.38, 9.41, 7.4, 6.39, 5.08, 3.65,
                          2.42, 1.454, 0.748, 1.299, 0.0766, 0.000185, 0.0001839])
power_first_2 = np.array([12.9, 11.5, 10.14, 8.7, 7.23, 6.2, 4.95, 4.21, 3.25, 2.22,
                          1.3, 0.783, 0.41, 0.7, 0.05, 0.000185, 0.00017])  # Moved KTP crystal away from Nd:YAG

# Uncertainties
power_first_error = 4.82e-1  # Uncertainty in power_first_1 and power_first_2
power_second_error = 2.21e-2  # Uncertainty in power_second

# Define a quadratic function
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

# Fit the data using curve_fit with constraints (a, b, c >= 0)
# Bounds for the coefficients (a, b, c) to ensure they stay >= 0
bounds = (0, np.inf)  # Lower bounds: 0, Upper bounds: infinity for all parameters

# Perform the curve fitting with constraints for the initial KTP position
params_1, cov_1 = curve_fit(quadratic, power_second, power_first_1, bounds=bounds)

# Perform the curve fitting with constraints for the moved KTP position
params_2, cov_2 = curve_fit(quadratic, power_second, power_first_2, bounds=bounds)

# Extract uncertainties (standard errors)
errors_1 = np.sqrt(np.diag(cov_1))
errors_2 = np.sqrt(np.diag(cov_2))

# Generate quadratic fit lines
fit_line_1 = quadratic(power_second, *params_1)
fit_line_2 = quadratic(power_second, *params_2)

# Plot data with error bars and quadratic fits
plt.figure(figsize=(10, 6))
plt.errorbar(power_second, power_first_1, xerr=power_second_error, yerr=power_first_error,
             fmt='o', label='Initial position of KTP', color='b')
plt.errorbar(power_second, power_first_2, xerr=power_second_error, yerr=power_first_error,
             fmt='s', label='Moved KTP crystal away from Nd:YAG', color='g')
plt.plot(power_second, fit_line_1, label=(
    f'Quadratic Fit (Initial KTP):\n'
    f'$y = ({params_1[0]:.2e} \\pm {errors_1[0]:.1e})x^2 + ({params_1[1]:.2e} \\pm {errors_1[1]:.1e})x + ({params_1[2]:.2e} \\pm {errors_1[2]:.1e})$'
), color='r')
plt.plot(power_second, fit_line_2, label=(
    f'Quadratic Fit (Moved KTP):\n'
    f'$y = ({params_2[0]:.2e} \\pm {errors_2[0]:.1e})x^2 + ({params_2[1]:.2e} \\pm {errors_2[1]:.1e})x + ({params_2[2]:.2e} \\pm {errors_2[2]:.1e})$'
), color='m')

plt.title("Quadratic Fit of Power-Power Data")
plt.ylabel("532nm Nd:YAG Laser Power [mW]")
plt.xlabel("1064nm Nd:YAG Laser Power [mW]")
plt.grid(True)
plt.legend()
plt.savefig("quadratic_power_fit.pdf")
plt.show()

# Print quadratic coefficients with uncertainties
print("Quadratic Fit for Initial KTP Position:")
print(f"y = ({params_1[0]:.4e} ± {errors_1[0]:.1e})x² + ({params_1[1]:.4e} ± {errors_1[1]:.1e})x + ({params_1[2]:.4e} ± {errors_1[2]:.1e})\n")

print("Quadratic Fit for Moved KTP:")
print(f"y = ({params_2[0]:.4e} ± {errors_2[0]:.1e})x² + ({params_2[1]:.4e} ± {errors_2[1]:.1e})x + ({params_2[2]:.4e} ± {errors_2[2]:.1e})")
