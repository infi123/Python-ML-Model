import numpy as np
import tensorflow as tf

# Load the preprocessed unseen data
data = np.load('data_preprocessed.npz')  # Replace with your actual preprocessed unseen data file
print(data.files)
# Extract the data to make predictions on
X_unseen = data['X_test']

# Load the trained model
model = tf.keras.models.load_model('house_price_model.h5')  # Replace with your actual model file if different

# Make predictions on the unseen data
predictions = model.predict(X_unseen)

# Print the predictions
print("Predictions: ", predictions)
