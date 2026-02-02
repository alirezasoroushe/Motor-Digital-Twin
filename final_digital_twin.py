import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# 1. Load trained Cloud Brain and data
model = load_model('motor_rul_model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
df = pd.read_csv('degradation_data.csv')

# 2. Prepare the data for prediction (Normalize)
scaler = MinMaxScaler()
vibration_scaled = scaler.fit_transform(df['vibration'].values.reshape(-1,1))

# 3. Predict RUL for every day to see the trend
# We use a 5-day sliding window
window = 5
predictions = []

for i in range(len(vibration_scaled) - window):
    seq = vibration_scaled[i:i+window].reshape(1, window, 1)
    pred = model.predict(seq, verbose=0)
    predictions.append(max(0, pred[0][0]))

# 4. Create the Dashboard
plt.figure(figsize=(12,6))

# Plot Actual Remaining Life vs AI Predicted Life
plt.plot(df['day'][window:], df['RUL'][window:], label='Actual Life', color='black', linestyle='--')
plt.plot(df['day'][window:], predictions, label='AI Predicted Life', color='blue', linewidth=2)

final_days = int(predictions[-1])
print(f"--- FINAL REPORT ---")
print(f"Current Operation Day: {df['day'].iloc[-1]}")
print(f"AI FORECAST: Motor will fail in approximately {final_days} days.")
print(f"ACTION REQUIRED: {'SCHEDULE MAINTENANCE' if final_days < 10 else 'MONITORING'}")
# Add a warning zone
plt.axhline(y=10, color='red', linestyle=':', label='Maintenance Threshold (10 Days)')
plt.fill_between(df['day'][window:], 0, 10, color='red', alpha=0.1)

plt.title('Industry 4.0 Digital Twin: Predictive Maintenance Dashboard')
plt.xlabel('Days in Operation')
plt.ylabel('Remaining Useful Life (Days)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

