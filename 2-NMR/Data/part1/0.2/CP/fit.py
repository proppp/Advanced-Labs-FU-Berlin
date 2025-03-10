
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# Load your specific data file
data = np.loadtxt("CP_0.0002s.dat")  # Update with the actual file path

# Extract columns
x = data[:, 0]  # Time (x-axis)
y = data[:, 1]  # Amplitude (y-axis)

# Shift all data vertically by 0.03
y_shifted = y - 1.03

# Define the initial mask to include all data
mask = x >= np.min(x)  # Mask the data to include the full range initially
x_filtered = x[mask]
y_filtered = y_shifted[mask]  # Use the shifted y-values

# Find initial peaks (this step happens only once)
peaks, _ = find_peaks(y_filtered, prominence=0.01)  # Initial prominence
selected_peaks = list(peaks)  # Store selected peaks

# Plot the data and initial peaks
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_filtered, y_filtered, color='blue', alpha=0.5, label='Data')  # Faded data
scatter_peaks = ax.scatter(x_filtered[peaks], y_filtered[peaks], color='red', marker='x', label='Detected Peaks')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Click to Select/Deselect Detected Peaks")
ax.legend()

# Store which peaks are selected/deselected
selected = []

# Exponential fit function: A * e^(-t/T_2)
def exp_decay(x, A, T_2):
    return A * np.exp(-x / T_2)

# Perform the fitting and save selected peaks to a text file
def fit_and_save_peaks():
    if len(selected_peaks) > 0:
        # Extract x and y values of the selected peaks
        valid_x = x_filtered[selected_peaks]
        valid_y = y_filtered[selected_peaks]

        # Perform exponential decay fit with initial guesses for A and T_2
        popt, pcov = curve_fit(exp_decay, valid_x, valid_y, p0=(139, 0.0092))

        # Extract fitting parameters and their uncertainties
        A, T_2 = popt
        perr = np.sqrt(np.diag(pcov))  # Uncertainties of the parameters
        A_err, T_2_err = perr

        # Print the fit parameters with uncertainties
        print(f"Fit parameters: A = {A:.4f} ± {A_err:.4f}, T_2 = {T_2:.4f} ± {T_2_err:.2e}")

        # Generate smooth x values for plotting the fit
        x_fit = np.linspace(valid_x.min(), valid_x.max(), 500)
        y_fit = exp_decay(x_fit, *popt)

        # Plot the exponential fit and pass the parameters to the plot function
        ax.plot(x_fit, y_fit, color='black', label=f'Fit: A={A:.4f}±{A_err:.4f}, T2={T_2:.4f}±{T_2_err:.2e}')

        # Add the fitting parameters to the plot as text
        fit_text = (f"A = {A:.4f} ± {A_err:.4f}\n"
                    f"T2 = {T_2:.4f} ± {T_2_err:.4f}")
        ax.text(0.05, 0.95, fit_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

        # Save selected peaks to a text file
        np.savetxt("selected_peaks.txt", np.array([valid_x, valid_y]).T, header="Time (s), Amplitude", fmt="%.6e")

        # Redraw the plot to include the fit curve
        fig.canvas.draw()

        # New plot showing the selected peaks and the fitting function
        plot_selected_peaks_and_fit(valid_x, valid_y, x_fit, y_fit, A, T_2, A_err, T_2_err)

        # Save the final plot as a PDF
        plt.savefig("selected_peaks_and_fit.pdf", format="pdf")
        print("Plot saved as 'selected_peaks_and_fit.pdf'.")

# Plot the selected peaks and the fitted function
def plot_selected_peaks_and_fit(valid_x, valid_y, x_fit, y_fit, A, T_2, A_err, T_2_err):
    # Create a new plot for the selected peaks and the exponential fit
    plt.figure(figsize=(10, 6))

    # Plot the data
    plt.plot(x_filtered, y_filtered, color='blue', alpha=0.5, label='Data')  # Faded data

    # Mark the selected peaks with 'x'
    plt.scatter(valid_x, valid_y, color='red', marker='x', label='Selected Peaks')

    # Plot the exponential fit and include the parameters in the label
    plt.plot(x_fit, y_fit, color='black', label=f'Fit: A={A:.4f}±{A_err:.4f}, T2={T_2:.4f}±{T_2_err:.4e}')

    # Customize the plot
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Selected Peaks and Exponential Fit")
    plt.legend()
    plt.grid(True)

    # Show the new plot
    plt.show()

# Callback function to toggle selected peaks
def on_click(event):
    global scatter_peaks  # Access the scatter object to update markers

    # Get the x-coordinate of the click
    x_click = event.xdata
    y_click = event.ydata

    if x_click is None or y_click is None:
        return  # Ignore if click is out of bounds

    # Find the closest peak to the click position (use absolute distance from x)
    closest_peak_idx = np.argmin(np.abs(x_filtered[peaks] - x_click))

    # Toggle peak selection
    peak_index = peaks[closest_peak_idx]

    if peak_index in selected_peaks:
        # Deselect peak if it's already selected
        selected_peaks.remove(peak_index)
    else:
        # Select peak if it's not already selected
        selected_peaks.append(peak_index)

    # Clear the previous scatter plot and plot the updated selected peaks
    scatter_peaks.remove()  # Remove the old scatter plot markers
    scatter_peaks = ax.scatter(x_filtered[peaks], y_filtered[peaks], color='blue', marker='o')  # Mark as deselected
    ax.scatter(x_filtered[selected_peaks], y_filtered[selected_peaks], color='red', marker='x')  # Mark as selected

    # Redraw the plot without performing the fit yet
    fig.canvas.draw()

# Callback function for pressing keys
def on_key(event):
    if event.key == 'enter':  # Check for Enter key press
        fit_and_save_peaks()

# Connect the callback functions
fig.canvas.mpl_connect('button_press_event', on_click)  # For peak selection/deselection
fig.canvas.mpl_connect('key_press_event', on_key)  # For Enter key press

# Dynamically set the x-limits based on the selected data
x_min = np.min(x_filtered)
x_max = np.max(x_filtered)

# Set the x-limits of the plot to a dynamic range
ax.set_xlim(x_min, x_max)

# Show the plot and let the user interact
plt.grid(True)
plt.show()
