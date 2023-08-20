import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error

# Load the preprocessed data
data = np.load('data_preprocessed.npz')

# Extract test data
X_test = data['X_test']
y_test = data['y_test']

# Load the trained model
model = tf.keras.models.load_model('house_price_model.h5')

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Root Mean Squared Error: {rmse}")
