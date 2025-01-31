
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def read_wsxm_cur(file_path):
    """
    Reads a WSxM .cur file, extracts IV curve data, and returns it as numpy arrays.
    """
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
                if len(values) == 4:  # Ensure the line contains 4 values
                    fb, fc, bb, bc = map(float, values)
                    forward_bias.append(fb)
                    forward_current.append(fc)
                    backward_bias.append(bb)
                    backward_current.append(bc)

    return (np.array(forward_bias), np.array(forward_current),
            np.array(backward_bias), np.array(backward_current))

def force_model(V_bias, a, V_CPD, c):
    """
    Model function: F_electr = a * (V_bias + V_CPD)^2 + c
    """
    return a * (V_bias + V_CPD) ** 2 + c

def fit_and_plot(V_bias, F_electr, label, color):
    """
    Fits the IV data and plots the result.
    """
    initial_guess = (1e-6, 0.0, 0.0)
    popt, _ = curve_fit(force_model, V_bias, F_electr, p0=initial_guess)

    a_fit, V_CPD_fit, c_fit = popt
    print(f"{label} Fit: a = {a_fit:.2e}, V_CPD = {V_CPD_fit:.3f}, c = {c_fit:.2e}")

    # Generate fitted curve
    V_fit = np.linspace(min(V_bias), max(V_bias), 200)
    F_fit = force_model(V_fit, *popt)

    # Plot
    plt.scatter(V_bias, F_electr, label=f"{label} Data", color=color, s=10)
    plt.plot(V_fit, F_fit, label=f"{label} Fit", color=color, linestyle="dashed")

    return popt  # Return fitted parameters

def analyze_cur_file(file_path):
    """
    Reads IV data, fits forward/backward separately, and also fits the averaged data.
    """
    forward_bias, forward_current, backward_bias, backward_current = read_wsxm_cur(file_path)

    plt.figure(figsize=(8, 6))

    # Fit forward scan
    fit_and_plot(forward_bias, forward_current, "Forward", "green")

    # Fit backward scan
    fit_and_plot(backward_bias, backward_current, "Backward", "red")

    # Compute and fit the average
    avg_bias = (forward_bias + backward_bias) / 2
    avg_current = (forward_current + backward_current) / 2
    fit_and_plot(avg_bias, avg_current, "Average", "blue")

    # Final plot settings
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("Force (A or Arbitrary Units)")
    plt.title("IV Curve Fitting: Forward, Backward & Average")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
file_path = "your_file.cur"  # Replace with actual file path
analyze_cur_file(file_path)
