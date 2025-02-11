
import numpy as np
import matplotlib.pyplot as plt

# Data (X and Y inverted)


power_second = np.array([336, 315.8, 292.4, 270.5, 250.1, 227.4, 202.5, 182.2, 159.7, 131.8,
                        106.7, 85.44, 64.3, 82.46, 33.5, 10.48, 6.63])
power_first_1 = np.array([17.2, 15.3, 13.15, 11.22, 10.38, 9.41, 7.4, 6.39, 5.08, 3.65,
                          2.42, 1.454, 0.748, 1.299, 0.0766, 0.000185, 0.0001839])  # Initial position of KTP
power_first_2 = np.array([12.9, 11.5, 10.14, 8.7, 7.23, 6.2, 4.95, 4.21, 3.25, 2.22,
                          1.3, 0.783, 0.41, 0.7, 0.05, 0.000185, 0.00017])  # Moved KTP crystal away from Nd:YAG

# Uncertainties
power_first_error = 4.82e-1  # Uncertainty in power_first_1 and power_first_2
power_second_error = 2.21e-2  # Uncertainty in power_second

# Quadratic fit (2nd-degree polynomial)
coeffs_1, cov_matrix_1 = np.polyfit(power_second, power_first_1, 2, cov=True)
coeffs_2, cov_matrix_2 = np.polyfit(power_second, power_first_2, 2, cov=True)

# Extract parameter uncertainties (standard errors from covariance matrix)
errors_1 = np.sqrt(np.diag(cov_matrix_1))
errors_2 = np.sqrt(np.diag(cov_matrix_2))

# Generate quadratic fit lines
fit_line_1 = np.polyval(coeffs_1, power_second)
fit_line_2 = np.polyval(coeffs_2, power_second)

# Plot data with error bars and quadratic fits
plt.figure(figsize=(10, 6))
plt.errorbar(power_second, power_first_1, xerr=power_second_error, yerr=power_first_error,
             fmt='o', label='Initial position of KTP', color='b')
plt.errorbar(power_second, power_first_2, xerr=power_second_error, yerr=power_first_error,
             fmt='s', label='Moved KTP crystal away from Nd:YAG', color='g')
plt.plot(power_second, fit_line_1, label=(
    f'Quadratic Fit (Initial KTP):\n'
    f'$y = ({coeffs_1[0]:.2e} \\pm {errors_1[0]:.1e})x^2 + ({coeffs_1[1]:.2e} \\pm {errors_1[1]:.1e})x + ({coeffs_1[2]:.2e} \\pm {errors_1[2]:.1e})$'
), color='r')
plt.plot(power_second, fit_line_2, label=(
    f'Quadratic Fit (Moved KTP):\n'
    f'$y = ({coeffs_2[0]:.2e} \\pm {errors_2[0]:.1e})x^2 + ({coeffs_2[1]:.2e} \\pm {errors_2[1]:.1e})x + ({coeffs_2[2]:.2e} \\pm {errors_2[2]:.1e})$'
), color='m')

plt.title("Quadratic Fit of Power-Power Data")
plt.xlabel("Nd:YAG Laser Power [mW]")
plt.ylabel("Diode Laser Power [mW]")
plt.grid(True)
plt.legend()
plt.savefig("quadratic_power_fit.pdf")
plt.show()

# Print quadratic coefficients with uncertainties
print("Quadratic Fit for Initial KTP Position:")
print(f"y = ({coeffs_1[0]:.4e} ± {errors_1[0]:.1e})x² + ({coeffs_1[1]:.4e} ± {errors_1[1]:.1e})x + ({coeffs_1[2]:.4e} ± {errors_1[2]:.1e})\n")

print("Quadratic Fit for Moved KTP:")
print(f"y = ({coeffs_2[0]:.4e} ± {errors_2[0]:.1e})x² + ({coeffs_2[1]:.4e} ± {errors_2[1]:.1e})x + ({coeffs_2[2]:.4e} ± {errors_2[2]:.1e})")
