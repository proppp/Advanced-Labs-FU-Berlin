import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

def read_wsxm_cur(file_path):
    """Reads a WSxM .cur file and extracts IV curve data."""
    data_started = False
    forward_bias, forward_current = [], []
    backward_bias, backward_current = [], []

    with open(file_path, 'r') as file:
        for line in file:
            if "[Header end]" in line:
                data_started = True
                continue
            if data_started:
                values = line.split()
                if len(values) == 4:
                    fb, fc, bb, bc = map(float, values)
                    forward_bias.append(fb)
                    forward_current.append(fc)
                    backward_bias.append(bb)
                    backward_current.append(bc)

    return np.array(forward_bias), np.array(forward_current), np.array(backward_bias), np.array(backward_current)

def force_model(V_bias, a, V_CPD, c):
    """Electrostatic force model: F_electr = a * (V_bias + V_CPD)^2 + c"""
    return a * (V_bias + V_CPD) ** 2 + c

def fit_iv_curve(V_bias, F_electr, sigma_F):
    """Fits the IV curve data considering uncertainties."""
    initial_guess = (1e-6, 0.0, 0.0)
    popt, pcov = curve_fit(force_model, V_bias, F_electr, p0=initial_guess, sigma=sigma_F, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan, np.nan, np.nan]
    return popt, perr

def format_params(params, uncert):
    """Formats parameters with uncertainties for legend."""
    return f"a={params[0]:.2e}±{uncert[0]:.2e}\nV_CPD={params[1]:.3f}±{uncert[1]:.3f}\nc={params[2]:.2e}±{uncert[2]:.2e}"

def plot_with_fit(title, x_data, y_data, y_err, fit_x, fit_y, fit_params, fit_uncert, file_name, color, label_data):
    """Generates plots with fit curves and displays fit parameters in the legend."""
    plt.figure(figsize=(8, 6))
    plt.errorbar(x_data, y_data, yerr=y_err, fmt='o', label=label_data, color=color, markersize=3, capsize=3)
    plt.plot(fit_x, fit_y, label=f"Fit: F_electr = a * (V_bias + V_CPD)^2 + c\n{format_params(fit_params, fit_uncert)}", color="black", linestyle="dashed")
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("Force (A or Arbitrary Units)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name)
    plt.show()

def process_folder(folder_path):
    """Processes all .cur files and generates separate plots for forward, backward, and averaged scans."""
    all_forward_bias, all_forward_current = [], []
    all_backward_bias, all_backward_current = [], []

    file_list = [f for f in os.listdir(folder_path) if f.endswith(".cur")]
    if not file_list:
        print("No .cur files found in the folder.")
        return

    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        forward_bias, forward_current, backward_bias, backward_current = read_wsxm_cur(file_path)
        all_forward_bias.append(forward_bias)
        all_forward_current.append(forward_current)
        all_backward_bias.append(backward_bias)
        all_backward_current.append(backward_current)

    all_forward_bias = np.array(all_forward_bias)
    all_forward_current = np.array(all_forward_current)
    all_backward_bias = np.array(all_backward_bias)
    all_backward_current = np.array(all_backward_current)

    avg_forward_bias = np.mean(all_forward_bias, axis=0)
    avg_forward_current = np.mean(all_forward_current, axis=0)
    std_forward_current = np.std(all_forward_current, axis=0)

    avg_backward_bias = np.mean(all_backward_bias, axis=0)
    avg_backward_current = np.mean(all_backward_current, axis=0)
    std_backward_current = np.std(all_backward_current, axis=0)

    avg_combined_bias = (avg_forward_bias + avg_backward_bias) / 2
    avg_combined_current = (avg_forward_current + avg_backward_current) / 2
    std_combined_current = np.sqrt(std_forward_current**2 + std_backward_current**2) / 2

    fit_params_forward, fit_uncert_forward = fit_iv_curve(avg_forward_bias, avg_forward_current, std_forward_current)
    fit_params_backward, fit_uncert_backward = fit_iv_curve(avg_backward_bias, avg_backward_current, std_backward_current)
    fit_params_combined, fit_uncert_combined = fit_iv_curve(avg_combined_bias, avg_combined_current, std_combined_current)

    V_fit = np.linspace(min(avg_forward_bias), max(avg_forward_bias), 200)
    F_fit_forward = force_model(V_fit, *fit_params_forward)
    F_fit_backward = force_model(V_fit, *fit_params_backward)
    F_fit_combined = force_model(V_fit, *fit_params_combined)

    print("\nFinal Fit Parameters with Uncertainties:")
    print(f"Forward Scan:  {format_params(fit_params_forward, fit_uncert_forward)}")
    print(f"Backward Scan: {format_params(fit_params_backward, fit_uncert_backward)}")
    print(f"Averaged Scan: {format_params(fit_params_combined, fit_uncert_combined)}")

    plot_with_fit("Forward IV Curve Fit", avg_forward_bias, avg_forward_current, std_forward_current,
                  V_fit, F_fit_forward, fit_params_forward, fit_uncert_forward, "task4Auforward.pdf", "blue", "Forward Data")

    plot_with_fit("Backward IV Curve Fit", avg_backward_bias, avg_backward_current, std_backward_current,
                  V_fit, F_fit_backward, fit_params_backward, fit_uncert_backward, "task4Aubackward.pdf", "red", "Backward Data")

    plot_with_fit("Averaged Forward & Backward IV Curve Fit", avg_combined_bias, avg_combined_current, std_combined_current,
                  V_fit, F_fit_combined, fit_params_combined, fit_uncert_combined, "task4Auavg.pdf", "purple", "Averaged Data")

# Example usage
folder_path = "."  # Replace with actual folder path
process_folder(folder_path)
