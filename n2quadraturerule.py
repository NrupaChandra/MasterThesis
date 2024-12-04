import numpy as np

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

# Main script to check the exact and computed integrals
def check_integrals():
    # Generate data
    coeffs, x0, nodes_weights, integrals = generate_data(num_samples=5)
    
    for i in range(len(coeffs)):
        a, b, c = coeffs[i]
        x0_i = x0[i, 0]
        nodes, weights = nodes_weights[i, :2], nodes_weights[i, 2:]

        # Exact integral
        exact = integrals[i][0]
        
        # Approximate integral using Gaussian quadrature
        approx_integral = gaussian_quadrature_integration(a, b, c, x0_i, nodes, weights)
        
        print(f"Sample {i+1}:")
        print(f"  Exact Integral: {exact:.4f}")
        print(f"  Computed Integral (from nodes and weights): {approx_integral:.4f}")
        print("--------------------------------------------------")

# Run the check
if __name__ == "__main__":
    check_integrals()
