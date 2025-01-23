
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Calibration constants
m = 0.842  # Slope in keV/channel
b = -48.806  # Intercept in keV

def gaussian(x, a, x0, sigma):
    """Gaussian function."""
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def parse_spectrum_file(filepath):
    """Parses a spectrum file to extract the spectrum data and measurement time."""
    channels = []
    counts = []
    measurement_time = None
    in_data_section = False

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()

            # Extract measurement time
            if line.startswith("$MEAS_TIM:"):
                try:
                    time_values = list(map(int, file.readline().strip().split()))
                    measurement_time = sum(time_values) / len(time_values)
                except Exception as e:
                    raise ValueError(f"Invalid format in $MEAS_TIM section: {e}")

            # Check for the start of the $DATA section
            if line.startswith("$DATA:"):
                in_data_section = True
                continue

            # Stop parsing if another section starts
            if in_data_section and line.startswith("$"):
                break

            # Process data lines
            if in_data_section:
                if len(channels) == 0:
                    # First line gives channel range
                    try:
                        start, end = map(int, line.split())
                        channels = list(range(start, end + 1))
                    except ValueError:
                        raise ValueError("Invalid channel range format in $DATA section.")
                else:
                    counts.extend(map(int, line.split()))

    if not channels or not counts or measurement_time is None:
        raise ValueError("No valid spectrum data or measurement time found in file.")

    # Scale counts by the measurement time
    scaled_counts = [count / measurement_time for count in counts]
    return channels, scaled_counts, measurement_time

def calibrate_channels_to_energy(channels):
    """Converts channel numbers to energy values (keV) using calibration constants."""
    return [m * channel + b for channel in channels]

def fit_gaussian_to_peak(energies, counts, fit_range):
    """Fits a Gaussian to the peak data within the specified energy range."""
    fit_indices = [i for i, energy in enumerate(energies) if fit_range[0] <= energy <= fit_range[1]]
    fit_energies = np.array(energies)[fit_indices]
    fit_counts = np.array(counts)[fit_indices]

    if len(fit_energies) == 0 or len(fit_counts) == 0:
        raise ValueError("No data points in the specified fit range.")

    # Initial guess for Gaussian parameters: amplitude, center, width
    p0 = [max(fit_counts), np.mean(fit_energies), 5]

    # Perform the Gaussian fit
    params, _ = curve_fit(gaussian, fit_energies, fit_counts, p0=p0)

    return params

def plot_sodium_spectra_with_peaks(folder_path, files, peak_ranges, vertical_lines):
    """Plots the spectra for sodium-22 detectors and identifies main peaks."""
    plt.figure(figsize=(12, 8))

    for file_name, ranges in zip(files, peak_ranges):
        try:
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_name}")

            # Parse and calibrate spectrum
            channels, scaled_counts, _ = parse_spectrum_file(file_path)
            energies = calibrate_channels_to_energy(channels)

            # Plot spectrum
            plt.plot(energies, scaled_counts, label=f"Spectrum: {file_name}", alpha=0.6)

            # Fit and annotate peaks
            for peak_range in ranges:
                params = fit_gaussian_to_peak(energies, scaled_counts, peak_range)
                fit_energies = np.linspace(peak_range[0], peak_range[1], 1000)
                fit_counts = gaussian(fit_energies, *params)

                plt.plot(fit_energies, fit_counts, linestyle='--', alpha=0.8, label=f"Fit: {file_name}, Peak @ {params[1]:.2f} keV")
                plt.annotate(f"{params[1]:.2f} keV", (params[1], max(fit_counts)), textcoords="offset points", xytext=(0, 10), ha='center')

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Add vertical lines at specified energies
    for energy in vertical_lines:
        plt.axvline(x=energy, color='red', linestyle='--', label=f'Compton edge @ {energy:.1f} keV')

    # Customize plot
    plt.title("Sodium-22 Spectrum and Compton Edge")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Scaled Counts (Counts per Second)")
    plt.legend()
    plt.grid()

    # Save plot as a PDF
    output_filename = "sodium_spectra_with_lines.pdf"
    plt.savefig(output_filename)
    print(f"Plot saved as {output_filename}")

    # Show the plot
    plt.show()

# Specify folder path, file names, and peak ranges
folder_path = "."
files = ["detector1_Na22.Spe", "detector2_Na22.Spe"]
peak_ranges = [
    [(400, 600), (1100, 1400)],  # Example ranges for Detector 1
    [(500, 550), (1200, 1300)]   # Example ranges for Detector 2
]
vertical_lines = [322.9, 996.5]  # Energies for vertical lines

# Plot the spectra and identify peaks
plot_sodium_spectra_with_peaks(folder_path, files, peak_ranges, vertical_lines)
