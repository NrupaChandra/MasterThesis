import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Exact integral of the quadratic function ax^2 + bx + c from 0 to x0
def exact_integral(a, b, c, x0):
    return (a * (x0**3) / 3) + (b * (x0**2) / 2) + (c * x0)

# Gaussian quadrature for n=2
def gaussian_quadrature_integration(a, b, c, x0, nodes, weights):
    # Calculate the function values at the nodes
    f_values = a * nodes**2 + b * nodes + c
    # Compute the approximate integral
    return np.sum(weights * f_values)

# Generate the data (coefficients and nodes/weights)
def generate_data(num_samples=100000):
    coeffs = np.random.uniform(-1, 1, size=(num_samples, 3))
    x0 = np.random.uniform(0, 1, size=(num_samples, 1))

    nodes_weights = []
    integrals = []

    for i in range(num_samples):
        a, b, c = coeffs[i]
        x0_i = x0[i, 0]
        
        # Exact integral calculation
        exact = exact_integral(a, b, c, x0_i)
        integrals.append(exact)

        # Gaussian quadrature nodes and weights for n=2
        t1 = -np.sqrt(1 / 3)
        t2 = np.sqrt(1 / 3)
        
        x1 = (x0_i / 2) * (t1 + 1)
        x2 = (x0_i / 2) * (t2 + 1)
        
        w1 = w2 = x0_i / 2
        
        nodes_weights.append([x1, x2, w1, w2])

    integrals = np.array(integrals, dtype=np.float64).reshape(-1, 1)
    nodes_weights = np.array(nodes_weights, dtype=np.float64)
    
    return coeffs, x0, nodes_weights, integrals

# Define a deeper model with additional layers and regularization
def build_cnn_model():
    model = models.Sequential()

    # Input Layer
    model.add(layers.InputLayer(input_shape=(4,)))  # 4 inputs: (a, b, c, x0)
    
    # Reshape for CNN (treat as 1D spatial data)
    model.add(layers.Reshape((4, 1)))  # Reshape input to (4, 1) for 1D CNN

    # Convolutional Layers with Batch Normalization
    model.add(layers.Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Conv1D(filters=256, kernel_size=2, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(filters=512, kernel_size=2, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Flatten())  # Flatten CNN output for Dense layers

    # Fully Connected (Dense) Layers
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout for regularization
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))

    # Output Layer
    model.add(layers.Dense(4))  # Predict 4 values (2 nodes, 2 weights)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
    
    return model

# Main script to train and evaluate the model
def train_and_evaluate():
    # Generate larger and normalized data
    coeffs, x0, nodes_weights, integrals = generate_data(num_samples=100000)
    inputs = np.hstack([coeffs, x0])  # Combine inputs
    outputs = nodes_weights          # Expected outputs
    scaler = StandardScaler()
    inputs = scaler.fit_transform(inputs)

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_cnn_model()

    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping])

    # Evaluate and compare
    for i in range(5):
        a, b, c = coeffs[i]
        x0_i = x0[i, 0]
        nodes, weights = nodes_weights[i, :2], nodes_weights[i, 2:]

        exact = integrals[i][0]
        approx_actual = gaussian_quadrature_integration(a, b, c, x0_i, nodes, weights)
        predicted = model.predict(scaler.transform([[a, b, c, x0_i]]))[0]
        approx_predicted = gaussian_quadrature_integration(a, b, c, x0_i, predicted[:2], predicted[2:])

        print(f"Sample {i+1}:")
        print(f"  Exact Integral: {exact:.8f}")
        print(f"  Computed Integral (from actual nodes/weights): {approx_actual:.8f}")
        print(f"  Computed Integral (from predicted nodes/weights): {approx_predicted:.8f}")
        print("--------------------------------------------------")

# Run the training and evaluation
if __name__ == "__main__":
    train_and_evaluate()
