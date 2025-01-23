
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
    """
    Parses a spectrum file to extract the spectrum data and measurement time.
    :param filepath: Path to the spectrum file.
    :return: (list of channels, list of scaled counts, average time in seconds)
    """
    channels = []
    counts = []
    measurement_time = None
    in_data_section = False

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()

            # Look for $MEAS_TIM to extract measurement time
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
                    # First line in $DATA: gives channel range
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
    """
    Calibrates the channel numbers to energy values (keV) using the linear calibration function.
    :param channels: List of channel numbers.
    :return: List of energy values in keV.
    """
    return [m * channel + b for channel in channels]

def fit_gaussian_to_spectrum(file_path, fit_range):
    """
    Fits a Gaussian to the spectrum data within a specified energy range.
    :param file_path: Path to the spectrum file.
    :param fit_range: Tuple specifying the energy range for Gaussian fitting (start, end).
    :return: Fitted Gaussian parameters (a, x0, sigma).
    """
    channels, scaled_counts, _ = parse_spectrum_file(file_path)
    energies = calibrate_channels_to_energy(channels)

    # Filter the data within the fitting range
    fit_indices = [i for i, energy in enumerate(energies) if fit_range[0] <= energy <= fit_range[1]]
    fit_energies = np.array(energies)[fit_indices]
    fit_counts = np.array(scaled_counts)[fit_indices]

    if len(fit_energies) == 0 or len(fit_counts) == 0:
        raise ValueError("No data points in the specified fit range.")

    # Initial guess for Gaussian parameters: amplitude, center, width
    p0 = [max(fit_counts), np.mean(fit_energies), 5]

    # Perform the Gaussian fit
    params, _ = curve_fit(gaussian, fit_energies, fit_counts, p0=p0)

    return params, fit_energies, fit_counts

def plot_all_spectra_with_fits(folder_path, fitting_ranges):
    """
    Plots all spectra and their Gaussian fits together in a single plot.
    :param folder_path: Path to the folder containing spectrum files.
    :param fitting_ranges: Dictionary of file names and their fitting ranges.
    """
    plt.figure(figsize=(12, 8))

    for file_name, fit_range in fitting_ranges.items():
        try:
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_name} with fit range: {fit_range}")

            # Parse and calibrate spectrum
            channels, scaled_counts, _ = parse_spectrum_file(file_path)
            energies = calibrate_channels_to_energy(channels)
            plt.plot(energies, scaled_counts, label=f"Spectrum: {file_name}")

            # Fit Gaussian and overlay
            params, fit_energies, _ = fit_gaussian_to_spectrum(file_path, fit_range)
            fit_counts = gaussian(fit_energies, *params)
            plt.plot(fit_energies, fit_counts, linestyle='--', label=f"Fit: {file_name}\nCenter={params[1]:.2f} keV")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    plt.title("Spectra and Gaussian Fits")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Scaled Counts (Counts per Second)")
    plt.legend()
    plt.grid()
    plt.show()


def plot_all_spectra_with_fits(folder_path, fitting_ranges):
    """
    Plots all spectra and their Gaussian fits together in a single plot, restricts range to 1000 keV, and saves as PDF.
    :param folder_path: Path to the folder containing spectrum files.
    :param fitting_ranges: Dictionary of file names and their fitting ranges.
    """
    plt.figure(figsize=(12, 8))

    for file_name, fit_range in fitting_ranges.items():
        try:
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_name} with fit range: {fit_range}")

            # Parse and calibrate spectrum
            channels, scaled_counts, _ = parse_spectrum_file(file_path)
            energies = calibrate_channels_to_energy(channels)
            plt.plot(energies, scaled_counts, label=f"Spectrum: {file_name}", alpha=0.45)

            # Fit Gaussian and overlay
            params, fit_energies, _ = fit_gaussian_to_spectrum(file_path, fit_range)
            fit_counts = gaussian(fit_energies, *params)
            plt.plot(fit_energies, fit_counts, linestyle='--', label=f"Fit: {file_name}\nCenter={params[1]:.2f} keV", alpha=1)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Limit x-axis range to 0–1000 keV
    plt.xlim(0, 1400)

    # Add titles, labels, and grid
    plt.title("Emitted Electron Spectra and Gaussian Fits")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Scaled Counts (Counts per Second)")
    plt.legend()
    plt.grid()

    # Save plot as a PDF
    output_filename = "spectra_fits_60deg.pdf"
    plt.savefig(output_filename)
    print(f"Plot saved as {output_filename}")

    # Display the plot
    plt.show()


# Specify the folder path and fitting ranges for each file
folder_path = "."
fitting_ranges = {
    "Na 120deg cut above": (0, 400),  # Example range in keV
    "Na 120deg cut below": (700, 1300)  # Example range in keV
}

# Plot all spectra and their Gaussian fits together
plot_all_spectra_with_fits(folder_path, fitting_ranges)
