import tensorflow as tf
import numpy as np

# Step 1: Build the Neural Network to Predict Nodes and Weights
def build_model():
    # Input layers
    coeff_input = tf.keras.Input(shape=(3,), name="coefficients")  # a, b, c
    x0_input = tf.keras.Input(shape=(1,), name="x0")  # X0
    
    # Process coefficients
    coeff_dense = tf.keras.layers.Dense(16, activation="relu")(coeff_input)

    # Process x0
    x0_dense = tf.keras.layers.Dense(16, activation="relu")(x0_input)
    
    # Combine coefficients and x0
    combined = tf.keras.layers.Concatenate()([coeff_dense, x0_dense])
    
    # Convolutional layers for processing
    cnn_input = tf.keras.layers.Reshape((1, -1, 1))(combined)  # Reshape for CNN compatibility
    conv1 = tf.keras.layers.Conv2D(32, (1, 3), activation="relu", padding="same")(cnn_input)
    conv2 = tf.keras.layers.Conv2D(64, (1, 3), activation="relu", padding="same")(conv1)
    conv3 = tf.keras.layers.Conv2D(128, (1, 3), activation="relu", padding="same")(conv2)
    flattened = tf.keras.layers.Flatten()(conv3)

    # Dense layers for prediction of nodes and weights
    dense1 = tf.keras.layers.Dense(512, activation="relu")(flattened)
    dense2 = tf.keras.layers.Dense(256, activation="relu")(dense1)
    dense3 = tf.keras.layers.Dense(128, activation="relu")(dense2)
    
    # Output layer for Gaussian quadrature nodes (2) and weights (2)
    output = tf.keras.layers.Dense(4, activation="linear")(dense3)  # 2 nodes and 2 weights
    
    # Model definition
    model = tf.keras.Model(inputs=[coeff_input, x0_input], outputs=output, name="QuadratureNet")
    return model

# Step 2: Generate Data for Training
def generate_data(num_samples=10000):
    coeffs = np.random.uniform(-1, 1, size=(num_samples, 3)).astype(np.float64)  # a, b, c
    x0 = np.random.uniform(0, 1, size=(num_samples, 1)).astype(np.float64)  # X0
    
    integrals = []
    for i in range(num_samples):
        a, b, c = coeffs[i]
        x0_i = x0[i, 0]
        integral = (
            a * (x0_i**3 / 3 - (0)**3 / 3) +
            b * (x0_i**2 / 2 - (0)**2 / 2) +
            c * (x0_i - 0)
        )
        integrals.append(integral)
    
    integrals = np.array(integrals, dtype=np.float64).reshape(-1, 1)
    
    # Normalize inputs
    coeffs = (coeffs - np.mean(coeffs, axis=0)) / np.std(coeffs, axis=0)
    x0 = (x0 - np.mean(x0, axis=0)) / np.std(x0, axis=0)
    
    return coeffs, x0, integrals

# Step 3: Gaussian Quadrature Rule
def gaussian_quadrature(nodes, weights, coeffs, x0):
    integral_estimate = 0
    for node, weight in zip(nodes, weights):
        # Calculate the integral estimate for each node-weight pair
        integral_estimate += weight * (coeffs[0] * node**3 + coeffs[1] * node**2 + coeffs[2] * node)
    return integral_estimate

# Step 4: Train the Model
def train_model():
    model = build_model()
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    coeffs, x0, integrals = generate_data()
    
    # Learning rate scheduler
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 0.95**epoch)
    
    model.fit([coeffs, x0], integrals, epochs=50, batch_size=32, callbacks=[lr_schedule])
    
    return model

# Step 5: Evaluate the Model
def evaluate_model(model, coeffs, x0, exact_integrals):
    # Get predictions (nodes and weights)
    predictions = model.predict([coeffs, x0])
    
    # Extract nodes and weights
    nodes = predictions[:, :2]
    weights = predictions[:, 2:]
    
    # Use Gaussian quadrature to estimate integrals
    estimated_integrals = np.array([gaussian_quadrature(nodes[i], weights[i], coeffs[i], x0[i]) for i in range(len(coeffs))])
    
    # Compare the estimated integrals with the exact integrals
    errors = np.abs(estimated_integrals - exact_integrals)
    mean_error = np.mean(errors)
    
    return mean_error

if __name__ == "__main__":
    model = train_model()
    
    # Evaluate on the same dataset used for training
    coeffs, x0, integrals = generate_data()
    
    # Evaluate the model performance
    error = evaluate_model(model, coeffs, x0, integrals)
    print(f"Mean error between estimated and exact integrals: {error}")
