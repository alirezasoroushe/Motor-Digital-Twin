# Intelligent Digital Twin for Motor Health Forecasting

**An End-to-End IIoT Predictive Maintenance System (Edge-to-Cloud)**

## 📌 Overview
This project implements a complete **Industrial IoT (IIoT) pipeline** for predictive maintenance. It simulates a "Digital Twin" of an industrial motor to generate realistic vibration data, processes it at the **Edge** for anomaly detection, and utilizes **Cloud-based Deep Learning (LSTM)** to forecast the Remaining Useful Life (RUL).

**Key Problem Solved:** Addressing "Data Scarcity" in smart manufacturing by generating high-fidelity synthetic data with non-stationary characteristics (load drift & noise) to train robust predictive models.

---

## 🚀 Key Engineering Features
* **Physics-Based Simulation:** Unlike simple sine waves, the generator now includes **Frequency Drift (49.5Hz - 50.5Hz)** to simulate real-world variable motor loads.
* **Edge Intelligence (The Gatekeeper):** Uses a **One-Class SVM** to filter normal data at the source, reducing cloud bandwidth usage by ~90%.
* **Robust MLOps:** Implements strict **Data Leakage Prevention** by serializing the `MinMaxScaler` during training and reloading the exact state for inference.
* **Signal Processing:** Applies **Moving Average Smoothing** on the dashboard to filter out sensor noise from the LSTM predictions.

---

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph L1 [Layer 1: Physical Digital Twin]
        A[data_generator.py] -->|Simulates Load Drift| B(Healthy Vibration)
        A -->|Injects 120Hz Harmonics| C(Faulty Vibration)
    end

    subgraph L2 [Layer 2: Edge Intelligence]
        B --> D[edge_detector.py]
        C --> D
        D --> E{One-Class SVM}
        E -- Normal --> F[Discard / Log Local]
        E -- Anomaly --> G[Trigger Cloud Upload]
    end

    subgraph L3 [Layer 3: Cloud Analytics]
        G --> H[cloud_train.py]
        H --> I[LSTM Neural Network]
        I -->|Save Model & Scaler| J[Artifact Store]
    end

    subgraph L4 [Layer 4: User Interface]
        J --> K[final_digital_twin.py]
        K -->|Smoothing Filter| L[Predictive Dashboard]
        L --> M{Maintenance Alert?}
        M -- "RUL < 10 Days" --> N[SCHEDULE MAINTENANCE]
        M -- "RUL > 10 Days" --> O[CONTINUE MONITORING]
    end

    %% Styling
    style N fill:#ff3333,color:#fff,stroke:#333,stroke-width:2px
    style E fill:#bbf,stroke:#333
    style I fill:#bbf,stroke:#333
    style L fill:#dfd,stroke:#333
