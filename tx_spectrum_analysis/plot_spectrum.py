import matplotlib.pyplot as plt
import pandas as pd

try:
    data = pd.read_csv("Measurement0018.csv")
    freq = data['Frequency [Hz]'][:]
    power = data['Maximum [dBm]'][:]
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)

freq = freq / 1e6  # Convert Hz to MHz
max_index = power.idxmax()
max_freq = freq[max_index]


plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(freq, power, label='Spectrum', linewidth=1)
ax.plot(max_freq, power[max_index], 'ro', label=f"Max Power: {power[max_index]:.2f} dBm")  # Mark the peak point
ax.set_xlabel('Frequency (MHz)', fontsize=10)
ax.set_ylabel('Power (dBm)', fontsize=10)
ax.set_title('Spectrum Analysis of TX', fontsize=12)
ax.grid(True, alpha=0.3)

plt.legend()
plt.tight_layout()
plt.show()