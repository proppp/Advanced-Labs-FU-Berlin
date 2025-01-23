import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import linregress



# Data

current = [500, 480, 457, 435, 414, 391, 366, 344, 322, 298, 273, 251, 231, 250, 200, 175, 170]

power = [336, 315.8, 292.4, 270.5, 250.1, 227.4, 202.5, 182.2, 159.7, 131.8, 106.7, 85.44, 64.3, 82.46, 33.5, 10.48, 6.63]




# Assume 1% relative error in current and power

current_errors = [0.01 * i for i in current]

power_errors = [0.01 * p for p in power]



# Perform linear regression

slope, intercept, r_value, p_value, std_err = linregress(current, power)



# Generate fit line

fit_line = [slope * i + intercept for i in current]



# Calculate residuals and chi-squared

residuals = np.array(power) - np.array(fit_line)

chi_squared = np.sum((residuals / power_errors) ** 2)

reduced_chi_squared = chi_squared / (len(current) - 2)  # degrees of freedom = n - 2



import numpy.linalg as LA



# Design matrix for linear fit

X = np.vstack([np.ones(len(current)), current]).T



# Weight matrix using 1% relative errors in power

weights = np.diag(1 / np.array(power_errors) ** 2)



# Covariance matrix of the fit parameters

cov_matrix = LA.inv(X.T @ weights @ X)



# Extract standard errors for slope and intercept

slope_error = np.sqrt(cov_matrix[1, 1])

intercept_error = np.sqrt(cov_matrix[0, 0])



slope, intercept, slope_error, intercept_error

# Update the plot to include errors in the fit parameters

plt.figure(figsize=(10, 6))

plt.errorbar(current, power, xerr=current_errors, yerr=power_errors, fmt='o', label='Data with errors', color='b')

plt.plot(current, fit_line,

         label=f'Linear fit: y = ({slope:.2f} ± {std_err:.2e})x + ({intercept:.2f} ± {intercept_error:.2e})',

         color='r')

plt.title("Linear Fit of Power vs Current")

plt.xlabel("Current (mA)")

plt.ylabel("Power (mW)")

plt.grid(True)

plt.legend()

plt.savefig("figure.pdf")
#plt.show()

# Plot data with error bars and the fit line

plt.figure(figsize=(10, 6))

plt.errorbar(current, power, xerr=current_errors, yerr=power_errors, fmt='o', label='Data', color='b')

#plt.plot(current, fit_line, label=f'Linear fit: y = {slope:.2f}x + {intercept:.2f}', color='r')

plt.title("Linear Fit of Power vs Current")

plt.xlabel("Current (mA)")

plt.ylabel("Power (mW)")

plt.grid(True)

plt.legend()

plt.show()



# Output slope, intercept, and reduced chi-squared

print(slope, intercept, std_err, reduced_chi_squared)
