import numpy as np
import matplotlib.pyplot as plt

# Time points: 6AM to 9PM -> 16 values (hours 6 to 21)
time_hours = np.arange(6, 22)

# Temperatures
june17 = np.array([71, 71, 72, 75, 78, 81, 82, 83, 85, 85, 85, 85, 84, 83, 83, 80])
june18 = np.array([68, 69, 70, 72, 74, 77, 78, 79, 81, 81, 84, 81, 79, 78, 77, 75])

# Part (a): FFT-based trigonometric interpolating polynomial
N = len(june17)
fft_coeffs = np.fft.fft(june17) / N  # Normalize FFT

# Interpolation grid (higher resolution for smooth curve)
interp_times = np.linspace(6, 21, 500)
interp_indices = (interp_times - 6) * N / (21 - 6)  # Map time to FFT sample index

# Reconstruct function from FFT coefficients
def trig_interp(x, coeffs):
    N = len(coeffs)
    result = np.zeros_like(x, dtype=np.complex128)
    for k in range(N):
        result += coeffs[k] * np.exp(2j * np.pi *k *x / N)
    return result.real

# Interpolated values for June 17 using FFT
june17_interp = trig_interp(interp_indices, fft_coeffs)

# Part (b): Plot results
plt.figure(figsize=(10, 6))
plt.plot(interp_times, june17_interp, label="FFT Interpolation (June 17)", color='blue')
plt.plot(time_hours, june18, 'ro-', label="June 18 Data")
plt.plot(time_hours, june17, 'kx', label="June 17 Original Points")

plt.xlabel("Hour of Day")
plt.ylabel("Temperature (°F)")
plt.title("Trigonometric Interpolation (FFT) of June 17 vs June 18 Temperatures")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


"""
Can the FFT interpolation of June 17 be used to predict June 18?

No, not really. June 18 follows a smoother trend. 
The June 17 FFT polynomial is too sensitive to individual values and boundary effects. 
Even though they peak at similar times, the blue line is not predictive beyond interpolation.
"""