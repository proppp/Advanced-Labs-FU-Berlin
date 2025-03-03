import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from PIL import Image
import os
import re
import scipy.special as sp

def hermite_gaussian_2d_squared(x, y, A, x0, y0, wx, wy, n, m):
    """Generates a squared 2D Hermite-Gaussian mode with specified n and m."""
    x_rot = x - x0
    y_rot = y - y0

    H_n_x = sp.hermite(n)(2 * x_rot / wx)
    H_m_y = sp.hermite(m)(2 * y_rot / wy)

    gaussian = np.exp(-(x_rot**2 / wx**2 + y_rot**2 / wy**2) / 2)
    hg = (A * H_n_x * H_m_y * gaussian)**2
    return hg

def fit_hermite_gaussian_squared(data_path, output_path, n, m):
    try:
        img = Image.open(data_path).convert('L')
        data = np.array(img)
        data_normalized = data / np.max(data)

        y, x = np.indices(data.shape)

        def error_function(params):
            A, x0, y0, wx, wy = params
            model = hermite_gaussian_2d_squared(x, y, A, x0, y0, wx, wy, n, m)
            residual = data_normalized - model
            huber_const = 1  # Adjust for robustness
            loss = np.where(np.abs(residual) <= huber_const, 0.5 * residual**2, huber_const * np.abs(residual) - 0.5 * huber_const**2).sum()
            return loss

        A_guess = 1.0
        max_index = np.unravel_index(np.argmax(data), data.shape)
        y0_guess, x0_guess = max_index
        wx_guess = data.shape[1] // 8
        wy_guess = data.shape[0] // 8
        initial_params = [A_guess, x0_guess, y0_guess, wx_guess, wy_guess]

        result = minimize(error_function, initial_params, method='L-BFGS-B', options={'maxiter': 5000})

        if result.success:
            A_fit, x0_fit, y0_fit, wx_fit, wy_fit = result.x
            A_fit = A_fit * np.max(data)

            fitted_model = hermite_gaussian_2d_squared(x, y, A_fit, x0_fit, y0_fit, wx_fit, wy_fit, n, m)
            error = data - fitted_model

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(data, cmap='gray')
            axes[0].set_title('Original Data')

            axes[1].imshow(fitted_model, cmap='gray')
            axes[1].set_title('Fitted Model')

            axes[2].imshow(error, cmap='gray')
            axes[2].set_title('Error (Data - Fit)')

            plt.tight_layout()

            ss_residual = np.sum(error**2)
            ss_total = np.sum((data - np.mean(data))**2)
            r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
            rmse = np.sqrt(ss_residual / data.size)

            # Add text with statistics and parameters
            text_str = f"n = {n}, m = {m}\n"
            text_str += f"A = {A_fit:.2f}, x0 = {x0_fit:.2f}, y0 = {y0_fit:.2f}\n"
            text_str += f"wx = {wx_fit:.2f}, wy = {wy_fit:.2f}\n"
            text_str += f"R-squared = {r_squared:.4f}\n"
            text_str += f"RMSE = {rmse:.4f}"

            plt.figtext(0.5, -0.1, text_str, ha="center", fontsize=10, bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})

            plt.savefig(output_path)
            plt.close(fig)

            return {"A": A_fit, "x0": x0_fit, "y0": y0_fit, "wx": wx_fit, "wy": wy_fit, "r_squared": r_squared, "rmse": rmse, "n": n, "m": m}
        else:
            print(f"Optimization failed for {data_path}")
            print(result)
            return None

    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Create the output directory if it doesn't exist
output_dir = "outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir("."):
    if filename.startswith("tem_") and filename.endswith(".bmp"):
        match = re.search(r"tem_(\d+)_(\d+)", filename)  # Extract n and m
        if match:
            n = int(match.group(1))
            m = int(match.group(2))
            print(f"Processing file: {filename} (n={n}, m={m})")
        else:
            print(f"Processing file: {filename}")
            n = 1  # Default values if n and m not found
            m = 2

        output_filename = os.path.join(output_dir, f"{filename[:-4]}_fit.png")
        fitted_params = fit_hermite_gaussian_squared(filename, output_filename, n, m)

        if fitted_params:
            print("Fitted parameters:", fitted_params)
            print(f"Fit results saved to {output_filename}")
        print("-" * 20)