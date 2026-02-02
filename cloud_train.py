import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. Load the "Digital Twin" degradation data
df = pd.read_csv('degradation_data.csv')
vibration_data = df['vibration'].values.reshape(-1,1)
rul_target = df['RUL'].values

# 2. Normalize data
scaler = MinMaxScaler()
vibration_scaled = scaler.fit_transform(vibration_data)

# 3. Prepare Sequences
# The model looks at 5 days of history to predict the future
X, y = [], []
window = 5
for i in range(len(vibration_scaled) - window):
    X.append(vibration_scaled[i:i+window])
    y.append(rul_target[i+window])

X, y = np.array(X), np.array(y)

# 4. Build the LSTM Architecture
model = Sequential([
    LSTM(50, activation ='relu', input_shape=(window, 1), return_sequences=False),
    Dropout(0.2), # Prevents the model from "memorizing" noise
    Dense(25, activation='relu'),
    Dense(1) # Final output: The predicted RUL (number of days)
])

model.compile(optimizer='adam', loss='mse')

# 5. Train the Model
print('Training Cloud Model (LSTM)...')
history = model.fit(X, y, epochs=100, batch_size=2, verbose=1)

# 6. Save the Brain
model.save('motor_rul_model.h5')
print("Model saved as 'motor_rul_model.h5'")

# Visualize Training Progress
plt.plot(history.history['loss'])
plt.title('Model Training Loss')
plt.show()