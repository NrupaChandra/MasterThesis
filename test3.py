import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
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
def generate_data(num_samples=10000):
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

# Define the Neural Network to predict nodes and weights
def build_nn_model():
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(4,)))  # 4 input values (a, b, c, x0)
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(4))  # Output 4 values (2 nodes, 2 weights)
    
    model.compile(optimizer='adam', loss='mse')
    return model

# Main script to train and evaluate the model
def train_and_evaluate():
    # Generate data
    coeffs, x0, nodes_weights, integrals = generate_data(num_samples=10000)
    
    # Prepare the input and output for the model
    inputs = np.hstack([coeffs, x0])  # Shape: (num_samples, 4) (a, b, c, x0)
    outputs = nodes_weights  # Shape: (num_samples, 4) (x1, x2, w1, w2)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(inputs, outputs, test_size=0.2, random_state=42)
    
    # Build and train the model
    model = build_nn_model()
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate the model
    for i in range(5):
        a, b, c = coeffs[i]
        x0_i = x0[i, 0]
        nodes, weights = nodes_weights[i, :2], nodes_weights[i, 2:]

        # Exact integral
        exact = integrals[i][0]
        
        # Approximate integral using Gaussian quadrature (actual nodes/weights)
        approx_integral_actual = gaussian_quadrature_integration(a, b, c, x0_i, nodes, weights)
        
        # Predict nodes and weights using the trained NN model
        predicted_nodes_weights = model.predict(np.array([[a, b, c, x0_i]]))[0]
        predicted_nodes = predicted_nodes_weights[:2]
        predicted_weights = predicted_nodes_weights[2:]
        
        # Approximate integral using predicted nodes/weights
        approx_integral_predicted = gaussian_quadrature_integration(a, b, c, x0_i, predicted_nodes, predicted_weights)
        
        print(f"Sample {i+1}:")
        print(f"  Exact Integral: {exact:.4f}")
        print(f"  Computed Integral (from actual nodes/weights): {approx_integral_actual:.4f}")
        print(f"  Computed Integral (from predicted nodes/weights): {approx_integral_predicted:.4f}")
        print("--------------------------------------------------")

# Run the training and evaluation
if __name__ == "__main__":
    train_and_evaluate()
