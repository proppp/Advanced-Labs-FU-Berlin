
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

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

def find_and_fit_peaks(channels, counts, search_ranges):
    """
    Finds peaks and fits Gaussians in specified search ranges.
    :param channels: List of channel numbers.
    :param counts: List of counts (scaled).
    :param search_ranges: List of tuples (start_channel, end_channel) defining the search ranges.
    :return: Fitted Gaussian parameters for each peak, and the peak positions.
    """
    # Convert to numpy arrays for easy manipulation
    channels = np.array(channels)
    counts = np.array(counts)

    fitted_params = []
    peak_positions = []

    for search_range in search_ranges:
        # Apply the search range
        start_idx = np.searchsorted(channels, search_range[0], side='left')
        end_idx = np.searchsorted(channels, search_range[1], side='right')
        search_channels = channels[start_idx:end_idx]
        search_counts = counts[start_idx:end_idx]

        # Find the highest peak in this range
        peaks, _ = find_peaks(search_counts, height=max(search_counts) * 0.1)  # Only significant peaks
        if len(peaks) == 0:
            continue
        highest_peak = peaks[np.argmax(search_counts[peaks])]  # Index of the highest peak in the range

        # Define a fitting range around the peak
        global_peak = start_idx + highest_peak  # Map to original indices
        fit_start = max(global_peak - 100, 0)
        fit_end = min(global_peak + 100, len(channels))

        # Fit a Gaussian to the data around the peak
        x_fit = channels[fit_start:fit_end]
        y_fit = counts[fit_start:fit_end]
        p0 = [max(y_fit), channels[global_peak], 5]  # Initial guess: [amplitude, center, width]
        try:
            params, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0)
            fitted_params.append(params)
            peak_positions.append(params[1])  # Center of the Gaussian
        except RuntimeError:
            print(f"Could not fit Gaussian for peak at channel {channels[global_peak]}.")

    return fitted_params, peak_positions


def plot_merged_spectra_with_peaks(folder_path, settings, material_names):
    """
    Reads spectrum files from two detectors for the same material, scales data, finds peaks, and plots them together.
    :param folder_path: Path to the folder containing spectrum files.
    :param settings: Dictionary with file-specific settings for peaks and ranges.
    :param material_names: Dictionary mapping file names to material names for plot titles.
    """
    # Organize files by material (based on shared suffix after detector)
    material_files = {}
    for file in os.listdir(folder_path):
        if not os.path.isfile(os.path.join(folder_path, file)):
            continue
        # Extract the material key (file name without "detector1_" or "detector2_")
        if file.startswith("detector1_"):
            key = file.replace("detector1_", "")
        elif file.startswith("detector2_"):
            key = file.replace("detector2_", "")
        else:
            continue
        material_files.setdefault(key, []).append(file)

    # Process each material
    for material_key, files in material_files.items():
        detector1_file = next((f for f in files if f.startswith("detector1_")), None)
        detector2_file = next((f for f in files if f.startswith("detector2_")), None)

        if not detector1_file or not detector2_file:
            print(f"Skipping {material_key}: Missing files for both detectors.")
            continue

        try:
            # Parse the spectra for both detectors
            detector1_path = os.path.join(folder_path, detector1_file)
            detector2_path = os.path.join(folder_path, detector2_file)

            channels1, scaled_counts1, _ = parse_spectrum_file(detector1_path)
            channels2, scaled_counts2, _ = parse_spectrum_file(detector2_path)

            # Find peaks and fit Gaussians for both detectors
            search_ranges = settings.get(detector1_file, {}).get("search_ranges", [])
            fitted_params1, peak_positions1 = find_and_fit_peaks(channels1, scaled_counts1, search_ranges)
            fitted_params2, peak_positions2 = find_and_fit_peaks(channels2, scaled_counts2, search_ranges)

            # Material name for the title
            material_name = material_names.get(detector1_file, material_key)

            # Create the plot
            plt.figure(figsize=(10, 10))

            # Plot Detector 1 spectrum
            plt.subplot(2, 1, 1)
            plt.plot(channels1, scaled_counts1, label="Spectrum (Detector 1)", color="blue")
            for params in fitted_params1:
                x_fit = np.linspace(min(channels1), max(channels1), 1000)
                y_fit = gaussian(x_fit, *params)
                plt.plot(x_fit, y_fit, label=f"Gaussian (center={params[1]:.1f})", linestyle="--")
            plt.title(f"Spectrum of {material_name} (Detectors 1 & 2), Channel vs. C.P.S.", fontsize=14)
            plt.ylabel("C.P.S.", fontsize=14)
            plt.legend()
            plt.grid()

            # Plot Detector 2 spectrum
            plt.subplot(2, 1, 2)
            plt.plot(channels2, scaled_counts2, label="Spectrum (Detector 2)", color="green")
            for params in fitted_params2:
                x_fit = np.linspace(min(channels2), max(channels2), 1000)
                y_fit = gaussian(x_fit, *params)
                plt.plot(x_fit, y_fit, label=f"Gaussian (center={params[1]:.1f})", linestyle="--")
            #plt.title(f"Spectrum of {material_name} (Detector 2)", fontsize=16)
            plt.xlabel("Channnel", fontsize=8)
            plt.ylabel("C.P.S", fontsize=14)
            plt.legend()
            plt.grid()

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error processing material {material_key}: {e}")


# Replace 'your_folder_path_here' with the path to your folder containing the spectrum files
#plot_merged_spectra_with_peaks(folder_path, settings, material_names)
# Define settings for each file
settings = {
    "detector1_Na22.Spe": {"search_ranges": [(500, 1000), (1300, 2000)]},
    "detector1_Am241.Spe": {"search_ranges": [(0, 200)]},
    "detector1_Cs137.Spe": {"search_ranges": [(650, 1050)]},
    "detector1_Co60.Spe": {"search_ranges": [(1238, 1536), (1546, 1789)]},
    "detector2_Na22.Spe": {"search_ranges": [(500, 1000), (1300, 2000)]},
    "detector2_Am241.Spe": {"search_ranges": [(0, 200)]},
    "detector2_Cs137.Spe": {"search_ranges": [(650, 1050)]},
    "detector2_Co60.Spe": {"search_ranges": [(1238, 1536), (1546, 1789)]},
    # Add more files and settings here
}

# Define material names for each file (LaTeX formatted)
material_names = {
    "detector1_Na22.Spe": r"$^{22}$Na",
    "detector1_Am241.Spe": r"$^{241}$Am",
    "detector1_Cs137.Spe": r"$^{137}$Cs",
    "detector1_Co60.Spe": r"$^{60}$Co",
    "detector2_Ba.Spe": r"$^{133}$Ba",
    "detector2_Cs137.Spe": r"$^{137}$Cs",
    "detector2_Am241.Spe": r"$^{241}$Am",
    "detector2_Co60.Spe": r"$^{60}$Co",
    "detector2_Na22.Spe": r"$^{22}$Na",
    # Add more mappings as necessary
}

# Replace 'your_folder_path_here' with the path to your folder containing the spectrum files
folder_path = "."
plot_merged_spectra_with_peaks(folder_path, settings, material_names)
