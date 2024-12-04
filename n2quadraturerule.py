import numpy as np

def gaussian_quadrature_integration(a, b, c, x0):
    # Define the Gaussian quadrature nodes and weights for n=2
    t1 = -1 / np.sqrt(3)
    t2 = 1 / np.sqrt(3)
    w1 = w2 = 1

    # Transform the nodes to the [0, x0] interval
    x1 = (x0 / 2) * (1 + t1)
    x2 = (x0 / 2) * (1 + t2)

    # Evaluate the function at the transformed nodes
    f_x1 = a * x1**2 + b * x1 + c
    f_x2 = a * x2**2 + b * x2 + c

    # Compute the integral using the Gaussian quadrature rule
    integral_approx = (x0 / 2) * (w1 * f_x1 + w2 * f_x2)

    return integral_approx

def exact_integral(a, b, c, x0):
    # Exact integral of ax^2 + bx + c from 0 to x0
    return (a * x0**3 / 3) + (b * x0**2 / 2) + (c * x0)

# Example: Define the quadratic function parameters (a, b, c)
a = 0
b = 0
c = 1
x0 = 1  # Upper limit of integration

# Calculate the exact integral
exact_val = exact_integral(a, b, c, x0)

# Calculate the integral using Gaussian quadrature
approx_val = gaussian_quadrature_integration(a, b, c, x0)

# Output results
print(f"Exact Integral: {exact_val:.4f}")
print(f"Approximated Integral (Gaussian Quadrature): {approx_val:.4f}")
