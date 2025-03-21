
import os
import numpy as np
import matplotlib.pyplot as plt

# Calibration constants
m = 0.842  # Slope in keV/channel
b = -48.806 # Intercept in keV

def parse_spectrum_file(filepath):
    """
    Parses a spectrum file to extract the spectrum data and measurement time.
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
    return channels, scaled_counts

def calibrate_channels_to_energy(channels):
    """
    Converts channel numbers to energy values (keV) using a linear calibration function.
    """
    return [m * channel + b for channel in channels]

def find_fwhm(energies, counts):
    """
    Calculates the Full Width at Half Maximum (FWHM) of the spectrum.
    """
    max_count = max(counts)
    half_max = max_count / 2

    # Find indices where counts cross half maximum
    above_half_max = np.where(np.array(counts) >= half_max)[0]

    if len(above_half_max) < 2:
        return None  # FWHM not found

    # Get corresponding energy values
    e_min, e_max = energies[above_half_max[0]], energies[above_half_max[-1]]

    return e_max - e_min

def plot_spectra_in_folder(folder_path):
    """
    Reads spectrum files in a folder, calibrates the energy, and plots their spectra.
    """
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    plt.figure(figsize=(10, 6))

    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            channels, scaled_counts = parse_spectrum_file(file_path)
            energies = calibrate_channels_to_energy(channels)
            fwhm = find_fwhm(energies, scaled_counts)

            # Format legend entry
            legend_label = f"{file} (FWHM: {fwhm:.2f} keV)" if fwhm else file

            plt.plot(energies, scaled_counts, label=legend_label)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    plt.title("Scaled and Calibrated Spectra with FWHM")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Scaled Counts (Counts per Second)")
    plt.legend()
    plt.grid()
    plt.show()

# Replace 'your_folder_path_here' with the path to your folder containing the spectrum files
folder_path = "."
plot_spectra_in_folder(folder_path)
