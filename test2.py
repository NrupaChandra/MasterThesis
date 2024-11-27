import tensorflow as tf
import numpy as np

# Step 1: Define a custom Lambda Layer
class LambdaFilterLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        coeffs, x0 = inputs
        return x0  # Pass through X0 as-is for now

# Step 2: Build the Neural Network
def build_model():
    # Input layers
    coeff_input = tf.keras.Input(shape=(3,), name="coefficients")  # a, b, c
    x0_input = tf.keras.Input(shape=(1,), name="x0")  # X0 (upper integration limit)
    
    # Lambda filter layer
    x0 = LambdaFilterLayer()([coeff_input, x0_input])
    
    # Reshape X0 for CNN
    x0_reshaped = tf.keras.layers.Reshape((1, 1, 1))(x0)
    
    # Convolutional layers to generate weights and nodes
    conv1 = tf.keras.layers.Conv2D(16, (1, 1), activation="relu")(x0_reshaped)
    conv2 = tf.keras.layers.Conv2D(32, (1, 1), activation="relu")(conv1)
    conv3 = tf.keras.layers.Conv2D(1, (1, 1), activation="linear")(conv2)  # Final output
    
    # Flatten to produce weights and nodes
    output = tf.keras.layers.Flatten()(conv3)
    
    # Model definition
    model = tf.keras.Model(inputs=[coeff_input, x0_input], outputs=output, name="QuadratureNet")
    return model

# Step 3: Generate Data for Training
def generate_data(num_samples=10000):
    coeffs = np.random.uniform(-1, 1, size=(num_samples, 3))  # Random a, b, c
    x0 = np.random.uniform(-1, 1, size=(num_samples, 1))  # Random X0 in range [-1, 1]
    integrals = []
    for i in range(num_samples):
        a, b, c = coeffs[i]
        x0_i = x0[i, 0]
        integral = (
            a * (x0_i**3 / 3 - (-1)**3 / 3) +
            b * (x0_i**2 / 2 - (-1)**2 / 2) +
            c * (x0_i - (-1))
        )  # Analytical integration
        integrals.append(integral)
    integrals = np.array(integrals).reshape(-1, 1)
    return coeffs, x0, integrals

# Step 4: Train the Model
def train_model():
    # Build the model
    model = build_model()
    
    # Compile the model
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    # Generate data
    coeffs, x0, integrals = generate_data()
    
    # Train the model
    model.fit([coeffs, x0], integrals, epochs=50, batch_size=32)
    
    return model

# Step 5: Evaluate the Model
model = train_model()
