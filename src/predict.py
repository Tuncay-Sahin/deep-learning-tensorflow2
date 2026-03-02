from pathlib import Path
import numpy as np
import joblib
import tensorflow as tf


# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCALER_PATH = PROJECT_ROOT / "data" / "processed" / "standard_scaler.joblib"
MODEL_PATH = PROJECT_ROOT / "reports" / "audiobook_model.keras"
THRESHOLD_PATH = PROJECT_ROOT / "reports" / "best_threshold.pkl"


# =====================================================
# Load Production Components
# =====================================================

print("Loading model components...")

scaler = joblib.load(SCALER_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

threshold = joblib.load(THRESHOLD_PATH)

print("Components loaded successfully ✅")


# =====================================================
# Example Incoming Customer Data
# =====================================================
# Feature count must match training data (10)

new_customer = np.array([
    [1, 0.8, 0.3, 1.2, 0.4, 0.9, 1.1, 0.2, 0.5, 0.7]
])


# =====================================================
# Preprocessing
# =====================================================

X_scaled = scaler.transform(new_customer)


# =====================================================
# Prediction
# =====================================================

probability = model.predict(X_scaled, verbose=0)[0][0]

prediction = int(probability >= threshold)


# =====================================================
# Output
# =====================================================

print("\nPrediction Result")
print("----------------------------")

print(f"Probability : {probability:.4f}")
print(f"Threshold   : {threshold}")
print(f"Prediction  : {prediction}")

if prediction == 1:
    print("Customer likely to repeat ✅")
else:
    print("Customer unlikely to repeat ❌")