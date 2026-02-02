import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt

# 1. Load the "Normal" data
data = pd.read_csv('normal_motor_data.csv')
X_train = data['vibration'].values.reshape(-1,1)

# 2. Train the "Gatekeeper" (One-Class SVM)
# 'nu' is the % of outliers we expect (0.01 = 1%)
model = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.01)
model.fit(X_train)
print('Edge Model Trained Successfully!')

# 3. Simulate a New Reading (Testing the model)
healthy_sample = np.array([[0.1]])
broken_sample = np.array([[3.5]])

# Predict: 1 = Normal, -1 = Anomaly
print(f"Prediction for 0.1: {model.predict(healthy_sample)}")
print(f"Prediction for 3.5: {model.predict(broken_sample)}")