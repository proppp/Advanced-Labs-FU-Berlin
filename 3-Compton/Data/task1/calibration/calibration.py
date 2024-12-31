
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define linear function for calibration
def linear_function(channel, m, b):
    return m * channel + b

# Energy table (keV) and isotopes
energy_table = {
    "$^{22}$Na": [511, 1274],
    "$^{60}$Co": [1173, 1332],
    "$^{137}$Cs": [661],
    "$^{241}$Am": [59],
}

# Peak positions from detectors
detector_data = {
    "Detector 1": {
        "$^{22}$Na": [677.86, 1548.63],
        "$^{60}$Co": [1464.99],
        "$^{137}$Cs": [853.69],
        "$^{241}$Am": [114.52],
    },
    "Detector 2": {
        "$^{22}$Na": [673.42, 1486.36],
        "$^{60}$Co": [1349.29, 1471.21],
        "$^{137}$Cs": [811.99],
        "$^{241}$Am": [115.45],
    },
}

# Perform calibration for each detector
calibration_results = {}
for detector, peaks in detector_data.items():
    measured_channels = []
    known_energies = []
    isotope_names = []

    # Match peaks to energies and store isotope names
    for isotope, channels in peaks.items():
        if isotope in energy_table:
            measured_channels.extend(channels)
            known_energies.extend(energy_table[isotope][:len(channels)])  # Match counts
            isotope_names.extend([isotope] * len(channels))  # Store isotope names

    # Perform linear fit
    popt, pcov = curve_fit(linear_function, measured_channels, known_energies)
    m, b = popt
    calibration_results[detector] = {"slope": m, "intercept": b}

    # Plot calibration
    plt.figure(figsize=(8, 6))
    plt.scatter(measured_channels, known_energies, color="blue", label="Data Points")

    # Calculate error bars (vertical distance from the line)
    y_fit = linear_function(np.array(measured_channels), m, b)
    errors = np.abs(np.array(known_energies) - y_fit)  # Vertical distance from the line

    # Plot vertical error bars
    plt.errorbar(measured_channels, known_energies, yerr=errors, fmt='o', color='blue', elinewidth=2, capsize=5)

    # Plot the linear fit
    x_fit = np.linspace(min(measured_channels), max(measured_channels), 100)
    y_fit_line = linear_function(x_fit, m, b)
    plt.plot(x_fit, y_fit_line, color="red", label=f"Fit: E = {m:.3f}C + {b:.3f}")

    # Annotate each point with the isotope name
    for i, (x, y, isotope) in enumerate(zip(measured_channels, known_energies, isotope_names)):
        plt.text(x, y + 30, isotope, fontsize=15, ha='center', color='black')

    # Calculate the average error
    avg_error = np.mean(errors)

    # Display the average error on the plot
    plt.text(0.5, 0.9, f"Avg Error: {avg_error:.2f} keV", transform=plt.gca().transAxes, fontsize=14, ha="center", va="center", bbox=dict(facecolor='white', edgecolor='none', alpha=0.75))

    # Set plot title
    plt.title(f"Energy Calibration for {detector}", fontsize=16)
    plt.xlabel("Channel", fontsize=14)
    plt.ylabel("Energy (keV)", fontsize=14)
    plt.legend()
    plt.grid()

    # Save plot as PDF
    plt.savefig(f"{detector}_calibration.pdf", format="pdf")
    plt.close()  # Close the figure to avoid memory issues

# Display calibration results
for detector, params in calibration_results.items():
    print(f"{detector} Calibration:")
    print(f"  Slope (m): {params['slope']:.3f} keV/channel")
    print(f"  Intercept (b): {params['intercept']:.3f} keV\n")
