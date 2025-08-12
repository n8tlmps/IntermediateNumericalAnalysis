import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([0., 0.15, 0.31, 0.5, 0.6, 0.75])
y = np.array([1.0, 1.004, 1.031, 1.117, 1.223, 1.422])

# Linear Fit (order 1) using normal equations
A_lin = np.array([
    [x.shape[0], np.sum(x)],
    [np.sum(x), np.sum(x**2)]
])
b_lin = np.array([
    np.sum(y),
    np.sum(x * y)
])
a_lin = np.linalg.solve(A_lin, b_lin)
y_lin_fit = a_lin[0] + a_lin[1] * x
print(f"Linear Fit: y = {a_lin[0]:.5f} + {a_lin[1]:.5f} * x")

# Polynomial fits using numpy.polyfit
coeffs_quad = np.polyfit(x, y, deg=2)
coeffs_cubic = np.polyfit(x, y, deg=3)

# Create polynomial functions
poly_quad = np.poly1d(coeffs_quad)
poly_cubic = np.poly1d(coeffs_cubic)

# Print coefficients
print("\nQuadratic Fit Coefficients (a2, a1, a0):", coeffs_quad)
print(poly_quad)

print("\nCubic Fit Coefficients (a3, a2, a1, a0):", coeffs_cubic)
print(poly_cubic)

# Generate smooth x values for plotting curves
x_fit = np.linspace(min(x), max(x), 300)

# Evaluate the polynomials
y_quad_fit = poly_quad(x_fit)
y_cubic_fit = poly_cubic(x_fit)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='black', label='Data points')
plt.plot(x, y_lin_fit, 'r--', label='Linear fit')
plt.plot(x_fit, y_quad_fit, 'g-', label='Quadratic fit')
plt.plot(x_fit, y_cubic_fit, 'b-', label='Cubic fit')

plt.title("Linear vs Quadratic vs Cubic Least Squares Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
