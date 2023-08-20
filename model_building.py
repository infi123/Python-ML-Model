import numpy as np
import tensorflow as tf

# Load the preprocessed data
data = np.load('data_preprocessed.npz')

# Extract training data
X_train = data['X_train']
y_train = data['y_train']

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Save the model
model.save('house_price_model.h5')
