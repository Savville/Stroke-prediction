import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import ctgan

# Set random seed for reproducibility
np.random.seed(42)

# Static data generation (1,000 patients)
n_samples = 1000
stroke_prob = 0.04  # 4% stroke prevalence
data = {
    'age': np.random.normal(50, 15, n_samples).clip(18, 90),
    'hypertension': np.random.binomial(1, 0.2, n_samples),
    'heart_disease': np.random.binomial(1, 0.1, n_samples),
    'bmi': np.random.normal(28, 5, n_samples).clip(15, 40),
    'glucose': np.random.normal(100, 20, n_samples).clip(70, 200),
    'smoking': np.random.binomial(1, 0.15, n_samples),
    'stroke': np.random.binomial(1, stroke_prob, n_samples)
}
static_df = pd.DataFrame(data)

# Use CTGAN to enhance correlations
ctgan_model = ctgan.CTGAN()
ctgan_model.fit(static_df)
static_df = ctgan_model.sample(n_samples)

# Time-series data generation
timesteps = 300
ts_data = []
for i in range(n_samples):
    is_stroke = static_df['stroke'].iloc[i]
    # PPG: HR, sys_bp, dia_bp, spo2
    if is_stroke:
        hr = np.random.normal(80, 10, timesteps).clip(50, 120) + np.linspace(0, 10, timesteps)  # Rising trend
        sys_bp = np.random.normal(130, 15, timesteps).clip(90, 200) + np.linspace(0, 20, timesteps)
        dia_bp = np.random.normal(85, 10, timesteps).clip(60, 120) + np.linspace(0, 10, timesteps)
        spo2 = np.random.normal(95, 2, timesteps).clip(90, 100) - np.linspace(0, 2, timesteps)  # Slight drop
    else:
        hr = np.random.normal(70, 5, timesteps).clip(50, 120)
        sys_bp = np.random.normal(120, 10, timesteps).clip(90, 200)
        dia_bp = np.random.normal(80, 5, timesteps).clip(60, 120)
        spo2 = np.random.normal(98, 1, timesteps).clip(90, 100)
    
    # ECG: RR intervals, QRS amplitude
    if is_stroke:
        rr = np.random.normal(0.8, 0.2, timesteps).clip(0.6, 1.2) * (1 + np.random.uniform(-0.1, 0.1, timesteps))  # Irregular for AF
        qrs = np.random.normal(1, 0.2, timesteps).clip(0.5, 2) - np.linspace(0, 0.1, timesteps)
    else:
        rr = np.random.normal(0.8, 0.05, timesteps).clip(0.6, 1.2)
        qrs = np.random.normal(1, 0.1, timesteps).clip(0.5, 2)
    
    # Accelerometer: x, y, z
    if is_stroke:
        acc_x = np.random.normal(0, 0.5, timesteps).clip(-2, 2) + np.random.uniform(-0.1, 0.1, timesteps)  # Asymmetry
        acc_y = np.random.normal(0, 0.5, timesteps).clip(-2, 2)
        acc_z = np.random.normal(0, 0.5, timesteps).clip(-2, 2)
    else:
        acc_x = np.random.normal(0, 0.3, timesteps).clip(-2, 2)
        acc_y = np.random.normal(0, 0.3, timesteps).clip(-2, 2)
        acc_z = np.random.normal(0, 0.3, timesteps).clip(-2, 2)
    
    ts_patient = np.stack([hr, sys_bp, dia_bp, spo2, rr, qrs, acc_x, acc_y, acc_z], axis=1)
    ts_data.append(ts_patient)

ts_data = np.array(ts_data)  # Shape: (1000, 300, 9)

# Save data
static_df.to_csv('synthetic_static_data.csv', index=False)
np.save('synthetic_ts_data.npy', ts_data)

# Example preprocessing
scaler = MinMaxScaler()
static_scaled = scaler.fit_transform(static_df.drop('stroke', axis=1))
ts_scaled = np.array([scaler.fit_transform(ts) for ts in ts_data])