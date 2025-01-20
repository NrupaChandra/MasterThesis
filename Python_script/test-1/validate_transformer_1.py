import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from model_transformer_1 import load_model
from dataloader_transformer_1 import PolynomialDataset
import utilities
from create_levelset import polynomial_functions

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

# Load the model
model = load_model('model_transformer_1_weights.pth').to(device)
model.eval()

# Load the validation data
dataset = PolynomialDataset('ValidateBernstein_p1_data.txt', 'ValidateBernstein_p1_output.txt')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Create the output folder if it doesn't exist
output_folder = r"C:\Git\MasterThesis\outputs"
os.makedirs(output_folder, exist_ok=True)

# Prepare lists to store results to compute error
true_nodes_x = []
true_nodes_y = []
true_weights = []

predicted_nodes_x = []
predicted_nodes_y = []
predicted_weights = []

# File to save predictions
output_file = os.path.join(output_folder, "predicted_data.txt")
with open(output_file, 'w') as f:
    # Write header
    f.write("number;id;nodes_x;nodes_y;weights\n")

with torch.no_grad():
    number = 1  # Initialize a counter for numbering predictions
    total_absolute_difference = 0.0  # Sum of absolute differences
    total_ids = 0  # Total number of IDs

    for exp_x, exp_y, coeff, true_values_x, true_values_y, true_values_w, id in dataloader:
        # Move inputs to the appropriate device (CPU or CUDA)
        exp_x, exp_y, coeff = (exp_x.to(device), exp_y.to(device), coeff.to(device))
        true_nodes_x = true_values_x.numpy()  # true values are for reference, not used directly in inference
        true_nodes_y = true_values_y.numpy()  # true values are for reference, not used directly in inference
        true_weights = true_values_w.numpy()  # true values are for reference, not used directly in inference

        # Run inference on the model
        [predicted_values_x, predicted_values_y, predicted_values_w] = model(exp_x, exp_y, coeff, None)

        # Process predictions
        predicted_nodes_x = predicted_values_x.cpu().numpy().flatten()
        predicted_nodes_y = predicted_values_y.cpu().numpy().flatten()
        predicted_weights = predicted_values_w.cpu().numpy().flatten()

        # Access the level set function using the ID
        poly_id = id[0]  # Access the first element of the ID tuple
        test_fn = polynomial_functions[poly_id]["function"]

        # Compute integrals using the level set function
        pred_integral = utilities.compute_integration(predicted_values_x, predicted_values_y, predicted_values_w, test_fn)
        true_integral = utilities.compute_integration(true_values_x, true_values_y, true_values_w, test_fn)

        # Compute the absolute difference for this ID
        absolute_difference = abs(pred_integral[0].item() - true_integral[0].item())
        total_absolute_difference += absolute_difference
        total_ids += 1

        # Print results
        print(f"Result of integration for {id}:")
        print(f"Algoim:  {true_integral[0].item():.4e}")
        print(f"QuadNET: {pred_integral[0].item():.4e}")
        print(f"Absolute Difference (r): {absolute_difference:.4e}")

        # Save predictions to text file
        with open(output_file, 'a') as f:
            f.write(
                f"{number};{id[0]};"  # Accessing the first element of the tuple
                f"{','.join(map(str, predicted_nodes_x))};"
                f"{','.join(map(str, predicted_nodes_y))};"
                f"{','.join(map(str, predicted_weights))}\n"
            )

        # Increment counter
        number += 1

    # Compute the accuracy number
    accuracyno = total_absolute_difference / total_ids if total_ids > 0 else 0
    print(f"Overall AccuracyNo: {accuracyno:.4e}")


        # Commenting out the plot generation
        # plt.figure(figsize=(10, 6))
        # plt.scatter(true_nodes_x, true_nodes_y, c=true_weights, cmap='viridis', label='True Points', alpha=0.6, marker='x')
        # plt.scatter(predicted_nodes_x, predicted_nodes_y, c=predicted_weights, cmap='plasma', label='Predicted Points', alpha=0.6)
        # plt.title('True vs Predicted Nodes')
        # plt.xlabel('X-coordinate')
        # plt.ylabel('Y-coordinate')
        # plt.colorbar(label='Weight (Coefficient)')
        # plt.legend()
        # plt.xlim(-1, 1)
        # plt.ylim(-1, 1)
        # plt.savefig(os.path.join(output_folder, f'{id[0]}.png'))
