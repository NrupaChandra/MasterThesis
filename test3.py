import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Function to calculate the integral using actual Gaussian quadrature nodes and weights
def gaussian_quadrature_integration(a, b, c, x0, nodes, weights):
    # Evaluate the function at the transformed nodes
    f_values = a * nodes**2 + b * nodes + c
    # Compute the integral using the Gaussian quadrature rule
    integral_approx = np.sum(weights * f_values)
    return integral_approx

# Function to calculate the exact integral of a quadratic function from 0 to x0
def exact_integral(a, b, c, x0):
    return (a * x0**3 / 3) + (b * x0**2 / 2) + (c * x0)

# Neural Network model to predict 2 nodes and 2 weights for Gaussian quadrature
def build_nn():
    model = models.Sequential()
    model.add(layers.Dense(64, input_dim=3, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(4))  # 2 nodes and 2 weights (output layer)
    model.compile(optimizer='adam', loss='mse')
    return model

# Training data generation (coefficients of quadratic function and actual nodes/weights)
def generate_data(num_samples=1000):
    coeffs = np.random.uniform(-1, 1, size=(num_samples, 3))  # Random coefficients a, b, c
    x0 = np.random.uniform(0, 1, size=(num_samples, 1))  # Random x0 values

    nodes_weights = []
    integrals = []

    for i in range(num_samples):
        a, b, c = coeffs[i]
        x0_i = x0[i, 0]

        # Exact integral of ax^2 + bx + c from 0 to x0
        exact_integral_val = exact_integral(a, b, c, x0_i)
        integrals.append(exact_integral_val)

        # Gaussian quadrature: n = 2 (nodes and weights)
        t1, t2 = -1 / np.sqrt(3), 1 / np.sqrt(3)
        w1, w2 = 1, 1

        x1 = (x0_i / 2) * (1 + t1)
        x2 = (x0_i / 2) * (1 + t2)
        nodes_weights.append([x1, x2, w1, w2])

    nodes_weights = np.array(nodes_weights)
    integrals = np.array(integrals).reshape(-1, 1)

    return coeffs, x0, nodes_weights, integrals

# Training the neural network
def train_nn():
    model = build_nn()

    # Generate training data
    coeffs, x0, nodes_weights, integrals = generate_data()

    # Prepare training inputs and outputs for NN
    X_train = coeffs
    y_train = nodes_weights

    # Train the neural network
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    return model, coeffs, x0, nodes_weights, integrals

# Evaluate the model and compare integrals
def evaluate_nn(model, coeffs, x0, nodes_weights, integrals):
    # Use the trained NN to predict nodes and weights
    predictions = model.predict(coeffs)

    # Calculate integrals for the exact, actual nodes/weights, and predicted nodes/weights
    exact_integrals = []
    actual_integrals = []
    predicted_integrals = []

    for i in range(len(coeffs)):
        a, b, c = coeffs[i]
        x0_i = x0[i, 0]

        # Exact integral
        exact_integrals.append(exact_integral(a, b, c, x0_i))

        # Actual integral using Gaussian quadrature nodes and weights
        nodes_actual, weights_actual = nodes_weights[i, :2], nodes_weights[i, 2:]
        actual_integrals.append(gaussian_quadrature_integration(a, b, c, x0_i, nodes_actual, weights_actual))

        # Predicted integral using NN predicted nodes and weights
        nodes_predicted, weights_predicted = predictions[i, :2], predictions[i, 2:]
        predicted_integrals.append(gaussian_quadrature_integration(a, b, c, x0_i, nodes_predicted, weights_predicted))

    # Print out the results for comparison
    for i in range(5):
        print(f"Sample {i+1}:")
        print(f"  Exact Integral: {exact_integrals[i]:.4f}")
        print(f"  Integral from Actual Nodes/Weights: {actual_integrals[i]:.4f}")
        print(f"  Integral from Predicted Nodes/Weights: {predicted_integrals[i]:.4f}")
        print("--------------------------------------------------")

# Main execution
if __name__ == "__main__":
    # Train the neural network and evaluate the results
    model, coeffs, x0, nodes_weights, integrals = train_nn()
    evaluate_nn(model, coeffs, x0, nodes_weights, integrals)
