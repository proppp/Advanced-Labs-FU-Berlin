
import os
import matplotlib.pyplot as plt

# Calibration constants
m = 0.842  # Slope in keV/channel
b = -48.806 # Intercept in keV

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

def plot_spectra_in_folder(folder_path):
    """
    Reads all spectrum files in a folder, scales their data, calibrates the energy, and plots their spectra.
    :param folder_path: Path to the folder containing spectrum files.
    """
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    plt.figure(figsize=(10, 6))

    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            channels, scaled_counts, avg_time = parse_spectrum_file(file_path)
            print(f"File: {file}, Average Time (seconds): {avg_time}")

            # Calibrate the channels to energy
            energies = calibrate_channels_to_energy(channels)

            # Plot the calibrated spectrum
            plt.plot(energies, scaled_counts, label=file)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    plt.title("Scaled and Calibrated Spectra from Folder")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Scaled Counts (Counts per Second)")
    plt.legend()
    plt.grid()
    plt.show()

# Replace 'your_folder_path_here' with the path to your folder containing the spectrum files
folder_path = "."
plot_spectra_in_folder(folder_path)
