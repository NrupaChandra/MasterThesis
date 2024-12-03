import tensorflow as tf
import numpy as np

# Step 1: Build the Neural Network
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
    
    # Add convolutional layers
    cnn_input = tf.keras.layers.Reshape((1, -1, 1))(combined)  # Reshape for CNN compatibility
    conv1 = tf.keras.layers.Conv2D(32, (1, 3), activation="relu", padding="same")(cnn_input)
    conv2 = tf.keras.layers.Conv2D(64, (1, 3), activation="relu", padding="same")(conv1)
    conv3 = tf.keras.layers.Conv2D(128, (1, 3), activation="relu", padding="same")(conv2)
    flattened = tf.keras.layers.Flatten()(conv3)

    # Dense layers for prediction
    dense1 = tf.keras.layers.Dense(512, activation="relu")(flattened)
    dense2 = tf.keras.layers.Dense(256, activation="relu")(dense1)
    dense3 = tf.keras.layers.Dense(128, activation="relu")(dense2)
    dense4 = tf.keras.layers.Dense(64, activation="relu")(dense3)
    
    # Output layer: 4 values (2 nodes and 2 weights)
    output = tf.keras.layers.Dense(4, activation="linear")(dense4)  # 4 outputs

    # Model definition
    model = tf.keras.Model(inputs=[coeff_input, x0_input], outputs=output, name="QuadratureNet")
    return model

# Step 2: Generate Data for Training
def generate_data(num_samples=10000):
    # Generate random coefficients (a, b, c) and x0
    coeffs = np.random.uniform(-1, 1, size=(num_samples, 3)).astype(np.float64)  # a, b, c
    x0 = np.random.uniform(0, 1, size=(num_samples, 1)).astype(np.float64)  # X0
    
    # Calculate analytical integrals
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

# Step 3: Implement Quadrature Rule and Loss Function
def quadrature_loss(y_true, y_pred):
    """
    Compute the loss using quadrature approximation:
    y_pred[0] = node1, y_pred[1] = node2, y_pred[2] = weight1, y_pred[3] = weight2
    """
    # Extract nodes and weights from the predicted output
    node1, node2, weight1, weight2 = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]
    
    # Apply quadrature rule: approximate integral = weight1 * f(node1) + weight2 * f(node2)
    # f(x) is the polynomial function evaluated at the quadrature nodes
    f_node1 = node1**3 + node2**2 + node1
    f_node2 = node2**3 + node1**2 + node2
    approx_integral = weight1 * f_node1 + weight2 * f_node2
    
    # Compute the Mean Squared Error between the exact and the approximate integral
    loss = tf.reduce_mean(tf.square(y_true - approx_integral))
    return loss

# Step 4: Train the Model
def train_model():
    # Build the model
    model = build_model()
    
    # Compile the model
    model.compile(optimizer="adam", loss=quadrature_loss, metrics=["mae"])
    
    # Generate data
    coeffs, x0, integrals = generate_data()
    
    # Learning rate scheduler
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 0.95**epoch)
    
    # Train the model
    model.fit([coeffs, x0], integrals, epochs=50, batch_size=32, callbacks=[lr_schedule])
    
    return model

# Step 5: Evaluate the Model
if __name__ == "__main__":
    model = train_model()
