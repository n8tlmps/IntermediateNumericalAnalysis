import numpy as np
import matplotlib.pyplot as plt

# Part (a): Data
months = np.arange(16)  # Entry indices 0 to 15
djia = np.array([
    14090, 14573, 14701, 15254, 14975, 15628,
    14834, 15193, 15616, 16009, 16441, 15373,
    16168, 16533, 16559, 16744
])

# FFT of the DJIA data
N = len(djia)
fft_coeffs = np.fft.fft(djia) / N

# Trigonometric interpolation function using FFT coefficients
def trig_interp(x, coeffs):
    N = len(coeffs)
    result = np.zeros_like(x, dtype=np.complex128)
    for k in range(N):
        result += coeffs[k] * np.exp(2j * np.pi * k * x / N)
    return result.real

# Part (b): Approximate April 8, 2013 and April 8, 2014
# Entry 1 is April 1, 2013 — April 8 is 1.25 (about a week later)
x_b = np.array([1.25, 13.25])
pred_b = trig_interp(x_b, fft_coeffs)

# Part (c): Actual vs predicted
actual_b = np.array([14613, 16256])
error_b = pred_b - actual_b

# Part (d): Approximate June 17, 2014 (about halfway through June)
x_d = np.array([15.5])
pred_d = trig_interp(x_d, fft_coeffs)[0]
actual_d = 16808

# Output results
print("=== Part (b) Approximations ===")
print(f"Predicted April 8, 2013 DJIA: {pred_b[0]:.2f}")
print(f"Predicted April 8, 2014 DJIA: {pred_b[1]:.2f}")

print("\n=== Part (c) Comparison to actual ===")
print(f"April 8, 2013 actual: {actual_b[0]}, error: {error_b[0]:.2f}")
print(f"April 8, 2014 actual: {actual_b[1]}, error: {error_b[1]:.2f}")

print("\n=== Part (d) June 17, 2014 Prediction ===")
print(f"Predicted June 17, 2014 DJIA: {pred_d:.2f}")
print(f"Actual June 17, 2014 DJIA: {actual_d}, error: {pred_d - actual_d:.2f}")

# Optional: Plot
x_dense = np.linspace(0, 15, 500)
djia_interp = trig_interp(x_dense, fft_coeffs)

plt.figure(figsize=(10, 6))
plt.plot(months, djia, 'ko', label='Original DJIA Data')
plt.plot(x_dense, djia_interp, 'b-', label='FFT Interpolation')
plt.plot(x_b, pred_b, 'ro', label='April 8 predictions')
plt.axvline(x=15.5, color='green', linestyle='--', label='June 17 prediction')

plt.title("Trigonometric Interpolation of DJIA (FFT)")
plt.xlabel("Entry (0 = Mar 2013 ... 15 = Jun 2014)")
plt.ylabel("DJIA Closing Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

"""
Trigonometric interpolation can capture trends smoothly.
However, prediction accuracy depends heavily on periodicity and whether the behavior is cyclical.
For finance data like DJIA, it's illustrative -- but real forecasting needs more sophisticated models (ARIMA, LSTM)
"""
