import tensorflow as tf
import numpy as np

# Step 1: Build the Neural Network with CNN
def build_model(): 
    coeff_input = tf.keras.Input(shape=(3,), name="coefficients")  # a, b, c
    x0_input = tf.keras.Input(shape=(1,), name="x0")  # Upper limit x0

    # Reshape inputs for CNN (add a channel dimension)
    coeff_input_reshaped = tf.keras.layers.Reshape((3, 1))(coeff_input)  # Reshaping for CNN
    x0_input_reshaped = tf.keras.layers.Reshape((1, 1))(x0_input)  # Reshaping for CNN

    # Apply CNN on coefficients input (1D convolution)
    coeff_cnn = tf.keras.layers.Conv1D(16, kernel_size=2, activation="relu")(coeff_input_reshaped)
    coeff_cnn = tf.keras.layers.BatchNormalization()(coeff_cnn)
    
    # Apply CNN on x0 input (1D convolution)
    x0_cnn = tf.keras.layers.Conv1D(16, kernel_size=1, activation="relu")(x0_input_reshaped)
    x0_cnn = tf.keras.layers.BatchNormalization()(x0_cnn)
    
    # Flatten CNN outputs
    coeff_flat = tf.keras.layers.Flatten()(coeff_cnn)
    x0_flat = tf.keras.layers.Flatten()(x0_cnn)
    
    # Combine flattened outputs
    combined = tf.keras.layers.Concatenate()([coeff_flat, x0_flat])
    
    # Fully connected layers
    dense1 = tf.keras.layers.Dense(128, activation="relu")(combined)
    dense1 = tf.keras.layers.BatchNormalization()(dense1)
    
    dense2 = tf.keras.layers.Dense(64, activation="relu")(dense1)
    dense2 = tf.keras.layers.BatchNormalization()(dense2)
    
    # Output 4 values: 2 nodes and 2 weights
    output = tf.keras.layers.Dense(4, activation="linear")(dense2)
    
    model = tf.keras.Model(inputs=[coeff_input, x0_input], outputs=output, name="GaussianQuadratureNet")
    return model

# Step 2: Generate Data
def generate_data(num_samples=10000):
    coeffs = np.random.uniform(-1, 1, size=(num_samples, 3)).astype(np.float64)
    x0 = np.random.uniform(0, 1, size=(num_samples, 1)).astype(np.float64)
    
    integrals = []
    nodes_weights = []
    for i in range(num_samples):
        a, b, c = coeffs[i]
        x0_i = x0[i, 0]
        
        exact_integral = (
            a * (x0_i**3 / 3) +
            b * (x0_i**2 / 2) +
            c * x0_i
        )
        integrals.append(exact_integral)
        
        t1 = -np.sqrt(1/3)
        t2 = np.sqrt(1/3)
        
        x1 = (x0_i / 2) * (t1 + 1)
        x2 = (x0_i / 2) * (t2 + 1)
        
        w1 = w2 = x0_i / 2
        
        nodes_weights.append([x1, x2, w1, w2])
    
    integrals = np.array(integrals, dtype=np.float64).reshape(-1, 1)
    nodes_weights = np.array(nodes_weights, dtype=np.float64)
    
    coeffs = (coeffs - np.mean(coeffs, axis=0)) / np.std(coeffs, axis=0)
    x0 = (x0 - np.mean(x0, axis=0)) / np.std(x0, axis=0)
    
    return coeffs, x0, nodes_weights, integrals

# Custom Callback to Track Integrals
class IntegralTrackingCallback(tf.keras.callbacks.Callback):
    def __init__(self, coeffs, x0, nodes_weights, integrals):
        super().__init__()
        self.coeffs = coeffs
        self.x0 = x0
        self.nodes_weights = nodes_weights
        self.integrals = integrals

    def on_epoch_end(self, epoch, logs=None):
        # Get predictions from the model on the training data
        predictions = self.model.predict([self.coeffs, self.x0])
        
        # Compute the integrals for both actual and predicted nodes/weights
        approx_integrals_predicted = []
        approx_integrals_actual = []

        for i in range(len(self.coeffs)):
            a, b, c = self.coeffs[i]
            x0_i = self.x0[i, 0]
            
            # Get the actual nodes/weights from the training data
            nodes_actual, weights_actual = self.nodes_weights[i][:2], self.nodes_weights[i][2:]
            f_values_actual = a * nodes_actual**2 + b * nodes_actual + c
            approx_integral_actual = np.sum(weights_actual * f_values_actual)
            approx_integrals_actual.append(approx_integral_actual)
            
            # Get the predicted nodes/weights from the model
            nodes_predicted, weights_predicted = predictions[i][:2], predictions[i][2:]
            f_values_predicted = a * nodes_predicted**2 + b * nodes_predicted + c
            approx_integral_predicted = np.sum(weights_predicted * f_values_predicted)
            approx_integrals_predicted.append(approx_integral_predicted)
        
        # Convert lists to numpy arrays
        approx_integrals_predicted = np.array(approx_integrals_predicted)
        approx_integrals_actual = np.array(approx_integrals_actual)
        
        # Print the comparison for the first 5 samples (or any number you like)
        for i in range(min(5, len(self.coeffs))):
            print(f"Epoch {epoch+1} - Sample {i+1}:")
            print(f"  Exact integral: {self.integrals[i][0]:.4f}")
            print(f"  Integral from actual nodes/weights: {approx_integrals_actual[i]:.4f}")
            print(f"  Integral from predicted nodes/weights: {approx_integrals_predicted[i]:.4f}")
            print("--------------------------------------------------")

# Step 3: Train the Model
def train_model():
    model = build_model()
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    coeffs, x0, nodes_weights, integrals = generate_data()
    
    val_split = 0.2
    train_size = int((1 - val_split) * len(coeffs))
    val_size = len(coeffs) - train_size

    X_train, X_val = coeffs[:train_size], coeffs[train_size:]
    y_train, y_val = nodes_weights[:train_size], nodes_weights[train_size:]

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 0.95**epoch)
    
    # Initialize the custom callback
    integral_callback = IntegralTrackingCallback(coeffs[:train_size], x0[:train_size], nodes_weights[:train_size], integrals[:train_size])

    # Train the model and include the custom callback
    model.fit([X_train, x0[:train_size]], y_train, 
              validation_data=([X_val, x0[train_size:]], y_val), 
              epochs=50, batch_size=32, 
              callbacks=[lr_schedule, integral_callback])
    
    return model, coeffs, x0, nodes_weights, integrals

# Step 4: Evaluate the Model
def evaluate_model(model, coeffs, x0, nodes_weights, integrals):
    predictions = model.predict([coeffs, x0])
    
    for i in range(5):
        print(f"Predicted nodes/weights: {predictions[i]}")
        print(f"Actual nodes/weights: {nodes_weights[i]}")
    
    approx_integrals = []
    for i in range(len(coeffs)):
        a, b, c = coeffs[i]
        x0_i = x0[i, 0]
        nodes, weights = predictions[i][:2], predictions[i][2:]
        f_values = a * nodes**2 + b * nodes + c
        approx_integral = np.sum(weights * f_values)
        approx_integrals.append(approx_integral)
    
    approx_integrals = np.array(approx_integrals, dtype=np.float64)
    for i in range(5):
        print(f"Approximated integral: {approx_integrals[i]}")
        print(f"Exact integral: {integrals[i][0]}")

# Main execution
if __name__ == "__main__":
    model, coeffs, x0, nodes_weights, integrals = train_model()
    evaluate_model(model, coeffs, x0, nodes_weights, integrals)
