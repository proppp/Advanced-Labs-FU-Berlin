
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_txt_files_in_folder(folder_path):
    # Get a list of all .txt files in the specified folder
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    if not txt_files:
        print("No .txt files found in the specified folder.")
        return

    for file_name in txt_files:
        file_path = os.path.join(folder_path, file_name)

        # Read the file and skip the header
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Extract header information
        header_info = {}
        for line in lines[:6]:  # The first six lines contain header info
            if ':' in line:  # Check if the line has a key-value pair
                key, value = line.split(':', 1)
                header_info[key.strip()] = value.strip()

        # Print header information for reference (optional)
        print(f"\nHeader Information for {file_name}:")
        for key, value in header_info.items():
            print(f"{key}: {value}")

        # Read the data section
        data = []
        for line in lines[6:]:  # Start from the 7th line onward
            if line.strip():  # Ensure line is not empty
                parts = line.split()
                if len(parts) == 2:  # We expect two values per line
                    try:
                        x_value = float(parts[0])
                        y_value = float(parts[1])
                        data.append((x_value, y_value))
                    except ValueError:
                        print(f"Skipping line due to conversion error: {line}")

        # Convert the data to a DataFrame for easier plotting
        if data:
            df = pd.DataFrame(data, columns=['X', 'Y'])

            # Plot the data
            plt.figure(figsize=(10, 6))
            plt.plot(df['X'], df['Y'], marker='o', linestyle='-', color='b')
            plt.title(f'Data Plot from {file_name}')
            plt.xlabel('X Values')
            plt.ylabel('Y Values')
            plt.grid()
            plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')  # Add a horizontal line at y=0 for reference
            plt.axvline(0, color='gray', linewidth=0.8, linestyle='--')  # Optional vertical line at x=0

            # Set x-axis limits based on the data
            plt.xlim(df['X'].min(), df['X'].max())  # Restrict x-axis range

            # Show the plot for the current file
            plt.show()
        else:
            print(f"No valid data found in {file_name}.")

# Example usage:
folder_path = '.'  # Replace with your actual folder path
plot_txt_files_in_folder(folder_path)
