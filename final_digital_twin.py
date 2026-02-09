import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib # <--- CRITICAL IMPORT
from tensorflow.keras.models import load_model

# 1. Load Resources (Model + Scaler)
model = load_model('motor_rul_model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
scaler = joblib.load('motor_scaler.gz')

# Load fresh data (In reality, this would be an API stream)
df = pd.read_csv('degradation_data.csv')

# 2. Preprocess 
# This ensures 0.5 vibration means the same thing here as it did during training.
vibration_scaled = scaler.transform(df['vibration'].values.reshape(-1,1))

# 3. Predict RUL
window = 5
predictions = []

for i in range(len(vibration_scaled) - window):
    seq = vibration_scaled[i:i+window].reshape(1, window, 1)
    pred = model.predict(seq, verbose=0)
    # Clamp prediction to 0 (cannot have negative days)
    predictions.append(max(0, pred[0][0]))

# 4. Smoothing for Dashboard (UX Improvement)
# Raw LSTM predictions can be jittery. We smooth them for the operator.
smooth_preds = pd.Series(predictions).rolling(window=3).mean()

# 5. Dashboard Visualization
plt.figure(figsize=(12,6))

# Plot Actual vs Predicted
plt.plot(df['day'][window:], df['RUL'][window:], label='Actual RUL', color='black', linestyle='--')
plt.plot(df['day'][window:], smooth_preds, label='AI Prediction (Smoothed)', color='blue', linewidth=2)

final_days = int(predictions[-1])
status = 'CRITICAL' if final_days < 10 else 'NORMAL'

print(f"--- DIGITAL TWIN STATUS ---")
print(f"Current Day: {df['day'].iloc[-1]}")
print(f"Predicted Remaining Life: {final_days} days")
print(f"Status: {status}")

# Warning Zone
plt.axhline(y=10, color='red', linestyle=':', label='Maintenance Threshold (10 Days)')
plt.fill_between(df['day'][window:], 0, 10, color='red', alpha=0.1)

plt.title(f'Digital Twin Dashboard - Status: {status}')
plt.xlabel('Days in Operation')
plt.ylabel('Remaining Useful Life (Days)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()