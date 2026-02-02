import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_motor_data(duration_sec=5, sampling_rate=1000, state='normal'):
    t = np.linspace(0, duration_sec, duration_sec*sampling_rate)

    # 1. Base Vibration (50Hz rotation)
    vibration = 0.5 * np.sin(2 * np.pi * 50 * t)

    # 2. Add Random Noise
    noise = np.random.normal(0, 0.1, len(t))
    vibration +=noise

    # 3. Inject Anomalies if state is 'faulty'
    if state == 'faulty':
        # Add high-frequency jitter and random high-amplitude spikes
        vibration += 0.8 * np.sin(2 * np.pi * 120 * t)
        spikes = np.random.choice([0,1], size=len(t), p=[0.99, 0.01]) * np.random.uniform(2, 4, len(t))
        vibration += spikes
    
    return t, vibration

# Generate both datasets
t, normal_vib = generate_motor_data(state='normal')
_, faulty_vib = generate_motor_data(state='faulty')

# Save to CSV for the next Phase
df = pd.DataFrame({'time': t, 'vibration': normal_vib})
df.to_csv('normal_motor_data.csv', index=False)
print("--- DATA SAVED SUCCESSFULLY ---")

# Plotting to see the difference
plt.figure(figsize=(12, 6))
plt.plot(t[:200], normal_vib[:200], label='Normal (Healthy)')
plt.plot(t[:200], faulty_vib[:200], label='Faulty (Anomaly)', alpha=0.7)
plt.title('Digital Twin: Motor Vibration Signatures')
plt.legend()
plt.show()

