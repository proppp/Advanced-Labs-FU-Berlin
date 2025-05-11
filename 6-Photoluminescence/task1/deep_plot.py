import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def read_ple_map(filename):
    """Reads a PLE map CSV file, handling various formatting issues."""
    df = pd.read_csv(filename)

    # Clean up column names (remove 'Unnamed' columns and convert to numeric)
    valid_cols = []
    emission_wavelengths = []
    for col in df.columns:
        if 'Unnamed' not in str(col):
            try:
                # Try to convert column name to float (for emission wavelength)
                val = float(col)
                emission_wavelengths.append(val)
                valid_cols.append(col)
            except ValueError:
                # Keep if it's the first column (excitation wavelength)
                if df.columns.get_loc(col) == 0:
                    valid_cols.append(col)

    # Extract data using only valid columns
    df = df[valid_cols]
    excitation_wavelengths = df.iloc[:, 0].values
    intensity_data = df.iloc[:, 1:].values

    # Convert emission wavelengths to numpy array
    emission_wavelengths = np.array(emission_wavelengths)

    return excitation_wavelengths, emission_wavelengths, intensity_data

def read_lamp_spectrum(filename):
    """Reads the lamp spectrum data file."""
    df = pd.read_csv(filename, delim_whitespace=True)
    wavelengths = df['Wavelength'].values
    intensities = df['R1'].values
    return wavelengths, intensities

def align_and_subtract(main_ex, main_em, main_int, bg_ex, bg_em, bg_int):
    """Aligns wavelengths and subtracts background, filling mismatches with zeros."""
    # Find overlapping excitation wavelengths
    ex_mask_main = np.isin(main_ex, bg_ex)
    ex_mask_bg = np.isin(bg_ex, main_ex)
    aligned_ex = main_ex[ex_mask_main]

    # Find overlapping emission wavelengths
    em_mask_main = np.isin(main_em, bg_em)
    em_mask_bg = np.isin(bg_em, main_em)
    aligned_em = main_em[em_mask_main]

    # Initialize corrected intensity with original data
    corrected_int = main_int.copy()

    # Only subtract where both datasets have data
    if np.any(ex_mask_main) and np.any(em_mask_main):
        corrected_int[np.ix_(ex_mask_main, em_mask_main)] = (
            main_int[np.ix_(ex_mask_main, em_mask_main)] -
            bg_int[np.ix_(ex_mask_bg, em_mask_bg)]
        )

    # Clip negative values to zero
    corrected_int = np.clip(corrected_int, 0, None)

    return main_ex, main_em, corrected_int  # Return original dimensions

def correct_for_lamp(em_wavelengths, lamp_wavelengths, lamp_intensities, corrected_int):
    """Corrects the PLE map for the lamp emission spectrum."""
    # Create interpolation function for lamp spectrum
    lamp_interp = interp1d(lamp_wavelengths, lamp_intensities,
                          bounds_error=False, fill_value=np.nan)

    # Interpolate lamp spectrum to match PLE emission wavelengths
    lamp_em = lamp_interp(em_wavelengths)

    # Replace NaN values with the nearest available value
    if np.isnan(lamp_em).any():
        # Find first and last non-NaN indices
        valid = ~np.isnan(lamp_em)
        if not np.any(valid):
            raise ValueError("No overlap between lamp spectrum and PLE emission wavelengths")

        # Forward fill then backward fill
        lamp_em = pd.Series(lamp_em).fillna(method='ffill').fillna(method='bfill').values

    # Avoid division by zero by replacing zeros with small value
    lamp_em[lamp_em == 0] = np.min(lamp_em[lamp_em > 0]) if np.any(lamp_em > 0) else 1e-10

    # Create 2D array of lamp spectrum (same shape as PLE map)
    lamp_2d = np.tile(lamp_em, (corrected_int.shape[0], 1))

    # Divide PLE map by lamp spectrum
    normalized_int = corrected_int / lamp_2d

    return normalized_int

# Read data
ex_main, em_main, int_main = read_ple_map('PLEMapOfTheMolecule.csv')
ex_bg, em_bg, int_bg = read_ple_map('PLEMapOfTheBlank.csv')

# Read lamp spectrum
lamp_wavelengths, lamp_intensities = read_lamp_spectrum('lamp.dat')

# Align and subtract
ex_final, em_final, int_corrected = align_and_subtract(
    ex_main, em_main, int_main, ex_bg, em_bg, int_bg
)

# Correct for lamp spectrum
int_normalized = correct_for_lamp(em_final, lamp_wavelengths, lamp_intensities, int_corrected)

# Plot
plt.figure(figsize=(12, 8))
plt.imshow(
    int_normalized,
    extent=[em_final.min(), em_final.max(),
            ex_final.min(), ex_final.max()],
    aspect='auto',
    cmap='viridis',
    origin='lower'
)
plt.colorbar(label='Normalized Intensity (a.u.)')
plt.xlabel('Emission Wavelength (nm)')
plt.ylabel('Excitation Wavelength (nm)')
plt.title('PLE Map (Background Corrected and Lamp Normalized)')

# Save the normalized data (optional)
normalized_df = pd.DataFrame(
    int_normalized,
    columns=em_final,
    index=ex_final
)
normalized_df.to_csv('PLEMap_Normalized.csv')

plt.tight_layout()
plt.savefig("2-molecule.pdf")
plt.show()
