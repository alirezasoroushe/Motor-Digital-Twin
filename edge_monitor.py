import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
import time

# 1. Setup: Load data and train the model
data = pd.read_csv('normal_motor_data.csv')
X_train = data['vibration'].values.reshape(-1,1)
model = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.01)
model.fit(X_train)

print("Edge Monitor Active. Monitoring Motor Vibration... ---")

# 2. Simulate the Real-Time Stream
test_stream = [0.12, 0.08, 0.15, 3.8, 0.11, 4.2, 0.09]

for i, reading in enumerate(test_stream):
    sample = np.array([[reading]])
    prediction = model.predict(sample)

    if prediction ==1:
        print(f"Sample {i}: Vibration {reading} -> [Status: OK]")
    else:
        print(f"ANOMALY DETECTED AT {reading}! Triggering Cloud Upload...")
        #Here we should trigger an MQTT message
        time.sleep(1) # Delay of sending data
    
    time.sleep(0.5) #Time between sensor readings