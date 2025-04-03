import torch
from dataloader_fnn import PolynomialDataset  # Import the dataset class
import numpy as np

# Function to construct polynomial functions
def construct_polynomial_functions_from_dataset(dataset):
    """
    Construct polynomial functions from a dataset.
    Each polynomial is stored in a dictionary with its ID as the key.
    """
    polynomials = {}
    for i in range(len(dataset)):
        # Get data from the dataset
        exp_x, exp_y, coeff, _, _, _, poly_id = dataset[i]

        # Convert tensors to numpy arrays
        exp_x = exp_x.numpy()
        exp_y = exp_y.numpy()
        coeff = coeff.numpy()

        # Construct the polynomial as a string
        terms = [f"({c}) * x**{ex} * y**{ey}" for c, ex, ey in zip(coeff, exp_x, exp_y)]
        polynomial_str = " + ".join(terms)

        # Create the lambda function from the string
        polynomial_fn = eval(f"lambda x, y: {polynomial_str}")

        # Store the function and string with its ID
        polynomials[poly_id] = {
            "function": polynomial_fn,
            "expression": polynomial_str
        }

    return polynomials

# Load the dataset
input_file = "TestBernstein_p1_data.txt"  
output_file = "TestBernstein_p1_output.txt"  
dataset = PolynomialDataset(input_file, output_file)

# Construct polynomial functions
polynomial_functions = construct_polynomial_functions_from_dataset(dataset)

# Example: Access and evaluate a polynomial
poly_id = list(polynomial_functions.keys())[0]  # Get the first polynomial ID
test_fn = polynomial_functions[poly_id]["function"]
poly_expression = polynomial_functions[poly_id]["expression"]

# Print the polynomial expression
print(f"Polynomial ID: {poly_id}")
print(f"Polynomial Expression: {poly_expression}")

