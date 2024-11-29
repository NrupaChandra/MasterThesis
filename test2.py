import tensorflow as tf
import numpy as np

# Step 1: Build the Neural Network
def build_model():
    # Input layers
    coeff_input = tf.keras.Input(shape=(3,), name="coefficients")  # a, b, c
    x0_input = tf.keras.Input(shape=(1,), name="x0")  # X0
    
    # Process coefficients
    #Rectified Linear Unit (ReLU) f(x) = max (0,x), makes it positive
    #This processes the coefficients and produces a 8-dimensional output.
    coeff_dense = tf.keras.layers.Dense(16, activation="relu")(coeff_input) 

    # Process x0
    # This processes the x0 and produces a 8-dimensional output.
    x0_dense = tf.keras.layers.Dense(16, activation="relu")(x0_input)
    
    # Combine coefficients and x0
    #Creates a single tensor combining the processed coefficients x0
    combined = tf.keras.layers.Concatenate()([coeff_dense, x0_dense])
    
    
    # Add convolutional layers to process the combined input
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
    dense5 = tf.keras.layers.Dense(32, activation="relu")(dense3)
    output = tf.keras.layers.Dense(1, activation="linear")(dense3)

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
    # Ensures the inputs are centered around 0 with a standard deviation of 1, which improves training stability.
    coeffs = (coeffs - np.mean(coeffs, axis=0)) / np.std(coeffs, axis=0)
    x0 = (x0 - np.mean(x0, axis=0)) / np.std(x0, axis=0)
    
    return coeffs, x0, integrals

# Step 3: Train the Model
def train_model():
    # Build the model
    model = build_model()
    
    # Compile the model
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    # Generate data
    coeffs, x0, integrals = generate_data()
    
    # Learning rate scheduler
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 0.95**epoch)
    
    # Train the model
    model.fit([coeffs, x0], integrals, epochs=50, batch_size=32, callbacks=[lr_schedule])
    
    return model

# Step 4: Evaluate the Model
if __name__ == "__main__":
    model = train_model()
