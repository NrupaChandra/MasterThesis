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
    f_values = a * nodes**2 + b * nodes + c
    return np.sum(weights * f_values)

# Generate the data (coefficients and nodes/weights)
def generate_data(num_samples=500000):
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

# Define a simpler neural network
def build_simple_nn_model(input_dim):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(4))  # Predict 4 values (2 nodes, 2 weights)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# Main script to train and evaluate the model
def train_and_evaluate_simpler():
    # Generate data
    coeffs, x0, nodes_weights, integrals = generate_data(num_samples=50000)
    inputs = np.hstack([coeffs, x0])  # Use only the basic features
    outputs = nodes_weights  # Target values: nodes and weights

    # Standardize inputs
    scaler = StandardScaler()
    inputs = scaler.fit_transform(inputs)

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    # Build simpler model
    model = build_simple_nn_model(input_dim=inputs.shape[1])

    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping])

    # Evaluate and compare results
    for i in range(5):
        a, b, c = coeffs[i]
        x0_i = x0[i, 0]
        nodes, weights = nodes_weights[i, :2], nodes_weights[i, 2:]

        exact = integrals[i][0]
        approx_actual = gaussian_quadrature_integration(a, b, c, x0_i, nodes, weights)

        input_sample = np.hstack([coeffs[i], x0[i]])
        input_sample = scaler.transform(input_sample.reshape(1, -1))

        predicted = model.predict(input_sample)[0]
        approx_predicted = gaussian_quadrature_integration(a, b, c, x0_i, predicted[:2], predicted[2:])

        print(f"Sample {i+1}:")
        print(f"  Exact Integral: {exact:.8f}")
        print(f"  Computed Integral (from actual nodes/weights): {approx_actual:.8f}")
        print(f"  Computed Integral (from predicted nodes/weights): {approx_predicted:.8f}")
        print("--------------------------------------------------")

# Run the training and evaluation
if __name__ == "__main__":
    train_and_evaluate_simpler()
