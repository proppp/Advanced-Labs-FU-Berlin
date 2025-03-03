
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

def read_wsxm_cur(file_path):
    """Reads a WSxM .cur file and extracts IV curve data."""
    data_started = False
    forward_bias, forward_sep = [], []
    backward_bias, backward_sep = [], []

    with open(file_path, 'r') as file:
        for line in file:
            if "[Header end]" in line:
                data_started = True
                continue
            if data_started:
                values = line.split()
                if len(values) == 4:
                    fb, fs, bb, bs = map(float, values)
                    forward_bias.append(fb)
                    forward_sep.append(fs)
                    backward_bias.append(bb)
                    backward_sep.append(bs)

    return np.array(forward_bias), np.array(forward_sep), np.array(backward_bias), np.array(backward_sep)

def linear_model(V_bias, m, b):
    """Linear model for fitting: y = m*x + b"""
    return m * V_bias + b

def fit_linear(V_bias, separation, sigma_sep, fit_range):
    """Fits a linear model within a specified range."""
    mask = (V_bias >= fit_range[0]) & (V_bias <= fit_range[1])
    V_fit = V_bias[mask]
    S_fit = separation[mask]
    sigma_fit = sigma_sep[mask]

    popt, pcov = curve_fit(linear_model, V_fit, S_fit, sigma=sigma_fit, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan, np.nan]

    return popt, perr

def process_folder(folder_path, fit_ranges, y_ranges):
    """Processes .cur files and performs linear fits on forward, backward, and averaged scans."""
    all_forward_bias, all_forward_sep = [], []
    all_backward_bias, all_backward_sep = [], []

    file_list = [f for f in os.listdir(folder_path) if f.endswith(".cur")]

    if not file_list:
        print("No .cur files found in the folder.")
        return

    print(f"Processing {len(file_list)} files...\n")

    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        forward_bias, forward_sep, backward_bias, backward_sep = read_wsxm_cur(file_path)
        all_forward_bias.append(forward_bias)
        all_forward_sep.append(forward_sep)
        all_backward_bias.append(backward_bias)
        all_backward_sep.append(backward_sep)

    # Convert to numpy arrays
    all_forward_bias = np.array(all_forward_bias)
    all_forward_sep = np.array(all_forward_sep)
    all_backward_bias = np.array(all_backward_bias)
    all_backward_sep = np.array(all_backward_sep)

    # Compute averages and uncertainties
    avg_forward_bias = np.mean(all_forward_bias, axis=0)
    avg_forward_sep = np.mean(all_forward_sep, axis=0)
    std_forward_sep = np.std(all_forward_sep, axis=0)

    avg_backward_bias = np.mean(all_backward_bias, axis=0)
    avg_backward_sep = np.mean(all_backward_sep, axis=0)
    std_backward_sep = np.std(all_backward_sep, axis=0)

    avg_combined_bias = (avg_forward_bias + avg_backward_bias) / 2
    avg_combined_sep = (avg_forward_sep + avg_backward_sep) / 2
    std_combined_sep = np.sqrt(std_forward_sep**2 + std_backward_sep**2) / 2  # Combined uncertainty

    # Fit linear models in the specified ranges
    fit_params_forward, fit_uncert_forward = fit_linear(avg_forward_bias, avg_forward_sep, std_forward_sep, fit_ranges["forward"])
    fit_params_backward, fit_uncert_backward = fit_linear(avg_backward_bias, avg_backward_sep, std_backward_sep, fit_ranges["backward"])
    fit_params_combined, fit_uncert_combined = fit_linear(avg_combined_bias, avg_combined_sep, std_combined_sep, fit_ranges["combined"])

    # Generate full-range fit lines
    V_fit_full = np.linspace(min(avg_forward_bias), max(avg_forward_bias), 200)
    S_fit_forward = linear_model(V_fit_full, *fit_params_forward)
    S_fit_backward = linear_model(V_fit_full, *fit_params_backward)
    S_fit_combined = linear_model(V_fit_full, *fit_params_combined)

    # Function to format fit parameters
    def format_params(params, uncert):
        return f"m = {params[0]:.3f} ± {uncert[0]:.3f}\nb = {params[1]:.3f} ± {uncert[1]:.3f}"

    # Plot Forward Scan
    plt.figure(figsize=(8, 6))
    plt.plot(V_fit_full, S_fit_forward, label=f"Linear Fit: y = mx + b\n{format_params(fit_params_forward, fit_uncert_forward)}",
             color="black", linestyle="dashed", linewidth=2, zorder=3)
    plt.errorbar(avg_forward_bias, avg_forward_sep, yerr=std_forward_sep, fmt='o', label="Forward Data",
                 color="orange", markersize=3, capsize=3, zorder=2, alpha = 0.95)
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("Tip-Sample Separation (nm)")
    plt.title("Forward IV Curve Linear Fit")
    plt.ylim(y_ranges["forward"])  # Set Y-axis range
    plt.legend()
    plt.grid(True)
    plt.savefig("forward.pdf")
    plt.show()

    # Plot Backward Scan
    plt.figure(figsize=(8, 6))
    plt.plot(V_fit_full, S_fit_backward, label=f"Linear Fit: y = mx + b\n{format_params(fit_params_backward, fit_uncert_backward)}",
             color="black", linestyle="dashed", linewidth=2, zorder=3)
    plt.errorbar(avg_backward_bias, avg_backward_sep, yerr=std_backward_sep, fmt='o', label="Backward Data",
                 color="red", markersize=3, capsize=3, zorder=2)
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("Tip-Sample Separation (nm)")
    plt.title("Backward IV Curve Linear Fit")
    plt.ylim(y_ranges["backward"])  # Set Y-axis range
    plt.legend()
    plt.grid(True)
    plt.savefig("backward.pdf")
    plt.show()

    # Plot Averaged Scan
    plt.figure(figsize=(8, 6))
    plt.plot(V_fit_full, S_fit_combined, label=f"Linear Fit\n{format_params(fit_params_combined, fit_uncert_combined)}",
             color="black", linestyle="dashed", linewidth=2, zorder=3)
    plt.errorbar(avg_combined_bias, avg_combined_sep, yerr=std_combined_sep, fmt='o', label="Averaged Data",
                 color="purple", markersize=3, capsize=3, zorder=2)
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("Tip-Sample Separation (nm)")
    plt.title("Averaged Forward & Backward IV Curve Linear Fit")
    plt.ylim(y_ranges["combined"])  # Set Y-axis range
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
folder_path = "."  # Replace with actual folder path
fit_ranges = {
    "forward": (-2, 2),  # Replace with the desired voltage range for linear fit
    "backward": (-2, 2),
    "combined": (-2, 2),
}
fit_ranges = {
    "forward": (-12, -8),  # Replace with the desired voltage range for linear fit
    "backward": (8, 12),
    "combined": (-2, 2),
}

y_ranges = {
    "forward": (-50, 100),  # Replace with desired y-axis range for forward scan
    "backward": (-100, 100),
    "combined": (0, 10),
}
process_folder(folder_path, fit_ranges, y_ranges)
