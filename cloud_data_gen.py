import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_degradation_data(days=100):
    # Simulate 100 days of data (1 reading per day)
    t = np.arange(days)

    # Exponential growth in vibration as parts wear out
    base_vibration = 0.5
    wear_trend = 0.05 * np.exp(0.05 * t)
    noise = np.random.normal(0, 0.05, days)

    total_vibration = base_vibration + wear_trend + noise

    # Calculate RUL(Remaining Useful Life): If day 50 is failure, day 0 has RUL of 50
    rul = days -t -1

    return pd.DataFrame({'day': t, 'vibration': total_vibration, 'RUL': rul})
df = generate_degradation_data()
df.to_csv('degradation_data.csv', index=False)

print('Cloud Training Data Saved: degradation_data.csv')

# Visualize the "Death Curve"
plt.figure(figsize=(10,5))
plt.plot(df['day'], df['vibration'], color='red', label='Vibration (Increasing)')
plt.ylabel('Vibration Intensity')
plt.xlabel('Days of Operation')
plt.title('Motor Degradation Over Time (Digital Twin Simulation)')
plt.legend()
plt.show()

