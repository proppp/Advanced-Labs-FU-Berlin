
import os
import numpy as np
import matplotlib.pyplot as plt

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

def plot_background_spectra(folder_path):
    """
    Reads background spectrum files, scales data, and plots them without fitting peaks.
    :param folder_path: Path to the folder containing spectrum files.
    """
    background_files = ['background1.Spe', 'background2.Spe']

    plt.figure(figsize=(10, 10))

    for idx, background_file in enumerate(background_files, start=1):
        try:
            # Parse the background spectrum file
            background_path = os.path.join(folder_path, background_file)
            channels, scaled_counts, _ = parse_spectrum_file(background_path)

            # Plot the background spectrum
            plt.subplot(2, 1, idx)
            plt.plot(channels, scaled_counts, label=f"Spectrum ({background_file})", color="blue")
            if idx == 1:
                # Remove x-axis label for the top plot
                plt.title("Background, Detector 1 and 2", fontsize=14)
                plt.ylabel("Scaled Counts (C.P.S.)", fontsize=12)
            else:
                plt.xlabel("Channel", fontsize=12)
                plt.ylabel("Scaled Counts (C.P.S.)", fontsize=12)
            plt.legend()
            plt.grid()

        except Exception as e:
            print(f"Error processing {background_file}: {e}")

    plt.tight_layout()
    plt.show()

# Define the folder path to your background files
folder_path = "."  # Adjust to the folder where your .Spe files are located
plot_background_spectra(folder_path)
