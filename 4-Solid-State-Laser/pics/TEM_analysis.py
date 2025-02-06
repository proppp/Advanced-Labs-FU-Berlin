import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy.optimize import curve_fit
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.measure import moments_central

# Define 2D Hermite-Gaussian function
def hermite_gaussian_2d(x, y, n, m, sx, sy, cx, cy, sigma=1.0):
    # Hermite polynomial scaling factors
    Hn = np.polynomial.hermite.hermval(x / sigma, [0] * n + [1])  # Hn(x)
    Hm = np.polynomial.hermite.hermval(y / sigma, [0] * m + [1])  # Hm(y)

    # Hermite-Gaussian mode (product of Gaussian envelope and Hermite polynomials)
    gauss = np.exp(-(x - cx)**2 / (2 * sx**2)) * np.exp(-(y - cy)**2 / (2 * sy**2))
    return Hn * Hm * gauss

# Fit function with scale & shift
def fit_function(coords, *params):
    x, y = coords
    num_coeffs = len(params) - 4  
    order = int(np.sqrt(num_coeffs)) - 1
    sx, sy, cx, cy = params[-4:]

    fitted = np.zeros_like(x)
    index = 0
    for n in range(order + 1):
        for m in range(order + 1):
            fitted += params[index] * hermite_gaussian_2d(x, y, n, m, sx, sy, cx, cy)
            index += 1
    return fitted.ravel()

# Load image
def load_image(filename):
    img = Image.open(filename).convert("L")  
    img_array = np.array(img, dtype=np.float64) / 255.0  
    return img_array

def estimate_scale_and_shift(image):
    ny, nx = image.shape
    y, x = np.mgrid[:ny, :nx]  

    total_intensity = np.sum(image)
    cx = np.sum(x * image) / total_intensity
    cy = np.sum(y * image) / total_intensity

    # Compute raw moments for size estimation
    mu20 = moments_central(image, center=(cy, cx), order=2)[0, 2]
    mu02 = moments_central(image, center=(cy, cx), order=2)[2, 0]

    # Adjusted priors based on raw moments and image dimensions
    prior_sx = np.sqrt(mu20) / nx
    prior_sy = np.sqrt(mu02) / ny
    prior_cx = nx * 0.45
    prior_cy = ny * 0.45

    # Blend estimated values with priors
    sx = 0.7 * (np.sqrt(mu20) / nx) + 0.3 * (prior_sx)  # Modified prior scaling
    sy = 0.7 * (np.sqrt(mu02) / ny) + 0.3 * (prior_sy)  # Modified prior scaling
    cx = 0.7 * (cx / nx) + 0.3 * (prior_cx / nx)  
    cy = 0.7 * (cy / ny) + 0.3 * (prior_cy / ny)  

    return sx, sy, cx, cy


# Fit function
def fit_hermite_gaussian(image, n, m):
    ny, nx = image.shape
    x = np.linspace(-1, 1, nx)  
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)

    sx, sy, cx, cy = estimate_scale_and_shift(image)

    initial_guess = np.ones((n + 1) * (m + 1))  
    initial_guess = np.append(initial_guess, [sx, sy, cx, cy])

    try:
        params, _ = curve_fit(fit_function, (X.ravel(), Y.ravel()), image.ravel(), 
                              p0=initial_guess, maxfev=10000)
        fitted_image = fit_function((X, Y), *params).reshape(ny, nx)
    except RuntimeError:
        print(f"⚠ Fit failed for (n={n}, m={m}), using fallback constant.")
        fitted_image = np.full_like(image, np.mean(image))  

    return fitted_image, (sx, sy, cx, cy)

# Create output directory
os.makedirs("outputs", exist_ok=True)

# Process images
for filename in os.listdir("."):
    match = re.match(r"tem_(\d+)(\d+)\.bmp", filename)
    if match:
        n, m = map(int, match.groups())  
        print(f"Processing {filename} with (n={n}, m={m})...")

        image = load_image(filename)
        fitted_image, (sx, sy, cx, cy) = fit_hermite_gaussian(image, n, m)

        print(f"Auto Fit Parameters: Scale (sx, sy) = ({sx:.3f}, {sy:.3f}), Shift (cx, cy) = ({cx:.3f}, {cy:.3f})")

        error_image = np.abs(image - fitted_image)
        mse = np.mean(error_image**2)
        max_error = np.max(error_image)
        ssim_score = ssim(image, fitted_image, data_range=1)

        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Original")

        axes[1].imshow(fitted_image, cmap="gray")
        axes[1].set_title(f"Fit (n={n}, m={m})")

        axes[2].imshow(error_image, cmap="hot")
        axes[2].set_title("Error")

        axes[3].text(0.1, 0.6, f"MSE: {mse:.5f}\nMax Error: {max_error:.3f}\nSSIM: {ssim_score:.3f}", fontsize=12)
        axes[3].set_xticks([])
        axes[3].set_yticks([])
        axes[3].set_frame_on(False)
        axes[3].set_title("Fit Stats")

        # Save each row
        plt.tight_layout()
        row_output_path = f"outputs/{filename.replace('.bmp', '_fit.png')}"
        plt.savefig(row_output_path)
        plt.close()

        print(f"✔ Saved: {row_output_path}")

print("✅ All images processed.")
