# src/quantize.py

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load trained model
model = joblib.load("model.joblib")

# Extract weights
coefs = model.coef_
intercept = model.intercept_

# Save unquantized weights
joblib.dump((coefs, intercept), "unquant_params.joblib")

# Manual quantization
min_val = coefs.min()
max_val = coefs.max()

scale = 255 / (max_val - min_val)
quantized_coefs = ((coefs - min_val) * scale).astype(np.uint8)
quantized_intercept = int((intercept - min_val) * scale)

# Save quantized parameters
joblib.dump((quantized_coefs, quantized_intercept, scale, min_val), "quant_params.joblib")

# ✅ Inference using dequantized weights
# Dequantize
dequant_coefs = quantized_coefs.astype(np.float32) / scale + min_val
dequant_intercept = quantized_intercept / scale + min_val

# Load sample input
data = fetch_california_housing()
X, y = data.data, data.target
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform inference with dequantized weights
y_pred = np.dot(X_test, dequant_coefs) + dequant_intercept
print("Sample prediction with dequantized model:", y_pred[:5])
