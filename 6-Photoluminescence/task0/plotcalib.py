
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('lamp.dat', delim_whitespace=True)

# Plot the data
plt.figure(figsize=(8, 5))
plt.plot(data['Wavelength'], data['R1'])
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (a.u.)')
plt.title('Lamp Emission Spectrum')
plt.grid(True)
plt.tight_layout()
plt.savefig("1-calibration.pdf")
plt.show()
