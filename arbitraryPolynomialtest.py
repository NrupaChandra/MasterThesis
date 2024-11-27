import numpy as np
from scipy.integrate import quad
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers # type: ignore

# Generate training data
N = 100000  # Number of training samples
max_degree = 4  # Maximum degree of the polynomial
coeffs = np.random.rand(N, max_degree + 1)  # Random coefficients
L = np.random.rand(N)  # Random lengths between 0 and 1
exact_integrals = np.zeros(N)

# Calculate exact integrals for each polynomial
for i in range(N):
    poly_coeff = coeffs[i, :]
    poly_func = lambda x: np.polyval(poly_coeff, x)
    exact_integrals[i], _ = quad(poly_func, 0, L[i])

# Prepare feature matrix
features = np.hstack((L.reshape(-1, 1), coeffs))

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, exact_integrals, test_size=0.2, random_state=42)

# Define the neural network using TensorFlow
def build_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(10, activation='relu'),
        # Add 9 Dense layers with ReLU activation
        *[layers.Dense(10, activation='relu') for _ in range(9)],
        layers.Dense(1)  # Output layer
    ])
    return model
# Initialize the model, loss function, and optimizer
input_dim = max_degree + 2  # Input size: max_degree + 2 (L + coefficients)
model = build_model(input_dim)
model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mse'])

# Convert data to TensorFlow tensors
X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
X_val_tensor = tf.convert_to_tensor(X_val, dtype=tf.float32)
y_val_tensor = tf.convert_to_tensor(y_val, dtype=tf.float32)

# Train the neural network
epochs = 100
history = model.fit(
    X_train_tensor, y_train_tensor,
    validation_data=(X_val_tensor, y_val_tensor),
    epochs=epochs,
    batch_size=32,
    verbose=1
)

# Validate the model
mse_val = model.evaluate(X_val_tensor, y_val_tensor, verbose=0)[1]
print(f"Validation MSE: {mse_val}")
