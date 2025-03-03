import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import hermite
import cv2

# Define the Hermite-Gaussian function to fit
# Includes shifts and scaling in x, y, and z directions
def model_function(x, y, m, n, a, b, c, x_shift, y_shift, z_offset, x_scale, y_scale):
    x = x_scale * (x - x_shift)  # Shift and scale x
    y = y_scale * (y - y_shift)  # Shift and scale y
    H_m = hermite(m)(x)  # Hermite polynomial of order m
    H_n = hermite(n)(y)  # Hermite polynomial of order n
    return (a * H_m * H_n * np.exp(-b * (x**2 + y**2)) + c) ** 2 + z_offset

# Load data from BMP file
def load_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    img = img.astype(np.float64) / 255.0  # Normalize to [0, 1]
    return img

z_data = load_image("./bmp/tem_22.bmp")
y_size, x_size = z_data.shape
x_data = np.linspace(-2, 2, x_size)
y_data = np.linspace(-2, 2, y_size)
x_mesh, y_mesh = np.meshgrid(x_data, y_data)

# Estimate initial parameters for better convergence
z_offset_init = np.min(z_data)
z_data_centered = z_data - z_offset_init
a_init = np.max(z_data_centered) ** 0.5
b_init = 1.0  # Initial guess for Gaussian decay factor
c_init = 0.0  # Initial guess for Hermite-Gaussian shift
x_shift_init, y_shift_init = np.mean(x_data), np.mean(y_data)
x_scale_init, y_scale_init = 1.0, 1.0

initial_params = [a_init, b_init, c_init, x_shift_init, y_shift_init, z_offset_init, x_scale_init, y_scale_init]

# Fit the model to the data
popt, pcov = curve_fit(lambda x, y, a, b, c, x_shift, y_shift, z_offset, x_scale, y_scale: model_function(x, y, 2, 2, a, b, c, x_shift, y_shift, z_offset, x_scale, y_scale),
                       (x_mesh, y_mesh), z_data, p0=initial_params)

# Extract the fitted parameters
a_fit, b_fit, c_fit, x_shift_fit, y_shift_fit, z_offset_fit, x_scale_fit, y_scale_fit = popt
print(f"Fitted parameters: a={a_fit:.2f}, b={b_fit:.2f}, c={c_fit:.2f}, x_shift={x_shift_fit:.2f}, y_shift={y_shift_fit:.2f}, z_offset={z_offset_fit:.2f}, x_scale={x_scale_fit:.2f}, y_scale={y_scale_fit:.2f}")

# Compute fitted function values
z_fit = model_function(x_mesh, y_mesh, 2, 2, *popt)

# Compute error map
error = np.abs(z_data - z_fit)

# Plot heatmaps
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot original data
c1 = axes[0].imshow(z_data, extent=[-2, 2, -2, 2], origin='lower', cmap='viridis')
axes[0].set_title('Original Data')
fig.colorbar(c1, ax=axes[0])

# Plot fitted function
c2 = axes[1].imshow(z_fit, extent=[-2, 2, -2, 2], origin='lower', cmap='viridis')
axes[1].set_title('Fitted Function')
fig.colorbar(c2, ax=axes[1])

# Plot error map
c3 = axes[2].imshow(error, extent=[-2, 2, -2, 2], origin='lower', cmap='inferno')
axes[2].set_title('Error Map')
fig.colorbar(c3, ax=axes[2])

plt.show()

# 3D plot of data and fit
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_mesh, y_mesh, z_data, label='Data')
ax.plot_surface(x_mesh, y_mesh, z_fit, color='red', alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Hermite-Gaussian Surface Fit (Squared, Shifted, Scaled)')
plt.show()
