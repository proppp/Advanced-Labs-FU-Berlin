
import os

def parse_spectrum_file(filepath):
    """
    Parses a spectrum file to extract the spectrum data and measurement time.
    :param filepath: Path to the spectrum file.
    :return: (list of channels, list of counts, measurement time)
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

    return channels, counts, measurement_time

def subtract_background(spectrum_filepath, background_counts):
    """
    Subtracts the background spectrum from a given spectrum file.
    :param spectrum_filepath: Path to the spectrum file.
    :param background_counts: List of counts from the background spectrum.
    :return: (modified channels, background-subtracted counts, measurement time)
    """
    channels, counts, measurement_time = parse_spectrum_file(spectrum_filepath)

    # Ensure the background and the spectrum have the same number of channels
    if len(counts) != len(background_counts):
        raise ValueError(f"Background and spectrum data have different lengths in file: {spectrum_filepath}")

    # Subtract background counts
    background_subtracted_counts = [max(0, count - background_counts[i]) for i, count in enumerate(counts)]

    return channels, background_subtracted_counts, measurement_time

def process_files_in_folder(folder_path, background_file):
    """
    Processes all .spe files in the folder by subtracting the background and saving the result.
    :param folder_path: Path to the folder containing spectrum files.
    :param background_file: Path to the background file.
    """
    # Parse the background file
    _, background_counts, _ = parse_spectrum_file(background_file)

    # Process each .spe file in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.Spe')]

    for file in files:
        spectrum_filepath = os.path.join(folder_path, file)
        try:
            # Subtract background
            channels, background_subtracted_counts, measurement_time = subtract_background(spectrum_filepath, background_counts)

            # Construct new file name
            new_filename = spectrum_filepath.replace(".Spe", "_backgroundsubtracted.spe")
            with open(new_filename, 'w') as new_file:
                with open(spectrum_filepath, 'r') as original_file:
                    # Copy the header until $DATA
                    in_data_section = False
                    for line in original_file:
                        line = line.strip()
                        if line.startswith("$DATA:"):
                            in_data_section = True
                            new_file.write(line + "\n")
                            break
                        else:
                            new_file.write(line + "\n")

                # Write the background-subtracted data
                for count in background_subtracted_counts:
                    new_file.write(f"       {count}\n")

            print(f"Processed and saved: {new_filename}")
        except Exception as e:
            print(f"Error processing file {file}: {e}")

# Replace these with your folder path and background file
folder_path = "."  # Your folder path here
background_file = os.path.join(folder_path, "background2.Spe")

process_files_in_folder(folder_path, background_file)
