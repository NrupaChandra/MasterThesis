#!/usr/bin/env python
import os
import shutil  # For copying files
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from model_hybrid import load_shallow_cnn_model
from multidataloader_fnn import MultiChunkDataset  
import utilities
import matplotlib.pyplot as plt

# Set device and default dtype (adjust as needed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

# Define directories for model and test results.
model_dir =  r"C:\Git\MasterThesis\Models\Hybrid\Hybrid_V5"
results_dir = r"C:\Git\RESULTS\Hybrid"
os.makedirs(results_dir, exist_ok=True)

# Redirect all console output to a log file
log_path = os.path.join(results_dir, "console_output.txt")
log_file = open(log_path, "w")
sys.stdout = log_file
sys.stderr = log_file

node_x_str = """-0.9602898564975362,-0.9602898564975362,-0.9602898564975362,-0.9602898564975362,-0.9602898564975362,-0.9602898564975362,-0.9602898564975362,-0.9602898564975362,
    -0.7966664774136267,-0.7966664774136267,-0.7966664774136267,-0.7966664774136267,-0.7966664774136267,-0.7966664774136267,-0.7966664774136267,-0.7966664774136267,
    -0.5255324099163290,-0.5255324099163290,-0.5255324099163290,-0.5255324099163290,-0.5255324099163290,-0.5255324099163290,-0.5255324099163290,-0.5255324099163290,
    -0.1834346424956498,-0.1834346424956498,-0.1834346424956498,-0.1834346424956498,-0.1834346424956498,-0.1834346424956498,-0.1834346424956498,-0.1834346424956498,
    0.1834346424956499,0.1834346424956499,0.1834346424956499,0.1834346424956499,0.1834346424956499,0.1834346424956499,0.1834346424956499,0.1834346424956499,
    0.5255324099163290,0.5255324099163290,0.5255324099163290,0.5255324099163290,0.5255324099163290,0.5255324099163290,0.5255324099163290,0.5255324099163290,
    0.7966664774136267,0.7966664774136267,0.7966664774136267,0.7966664774136267,0.7966664774136267,0.7966664774136267,0.7966664774136267,0.7966664774136267,
    0.9602898564975362,0.9602898564975362,0.9602898564975362,0.9602898564975362,0.9602898564975362,0.9602898564975362,0.9602898564975362,0.9602898564975362"""

node_y_str = """-0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956499,0.5255324099163290,0.7966664774136267,0.9602898564975362,
    -0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956499,0.5255324099163290,0.7966664774136267,0.9602898564975362,
    -0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956499,0.5255324099163290,0.7966664774136267,0.9602898564975362,
    -0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956499,0.5255324099163290,0.7966664774136267,0.9602898564975362,
    -0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956499,0.5255324099163290,0.7966664774136267,0.9602898564975362,
    -0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956499,0.5255324099163290,0.7966664774136267,0.9602898564975362,
    -0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956499,0.5255324099163290,0.7966664774136267,0.9602898564975362,
    -0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956499,0.5255324099163290,0.7966664774136267,0.9602898564975362"""

# Load the model from the specified directory and set it to evaluation mode.
model_path = os.path.join(model_dir, 'fnn_model_weights_v4.pth')
model = load_shallow_cnn_model(
    model_path,
    node_x_str=node_x_str,
    node_y_str=node_y_str
).to(device)
model.eval()

# Define a test function for integration.
def test_fn(x, y):
    return 1

# Define the data directory.
data_dir = r"C:\Git\Data"

# Create dataset using the base directory.
dataset = MultiChunkDataset(
    index_file=os.path.join(data_dir, 'combined_preprocessed_chunks_TestBernstein/index.txt'),
    base_dir=data_dir
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Create the output folder for results.
output_folder = results_dir
os.makedirs(output_folder, exist_ok=True)

# File to save predictions.
output_file = os.path.join(output_folder, "predicted_data_fnn.txt")
with open(output_file, 'w') as f:
    # Write header.
    f.write("number;id;nodes_x;nodes_y;weights\n")

# --- START OF ADDITIONS FOR MSE ---
total_squared_difference = 0.0  # Sum of squared differences for MSE.
# --- END OF ADDITIONS FOR MSE ---

# Prepare accumulators for error computations and to store integrals.
total_absolute_difference = 0.0  # Sum of absolute differences for MAE.
total_ids = 0                    # Total number of samples.
predicted_integrals = []         # List to store predicted integrals.
true_integrals = []              # List to store true integrals.
relative_errors = []             # List to store relative error for each sample.
rel_error_info = []              # List to store (sample_id, relative_error) for outlier identification.
number = 1                       # Counter for numbering predictions.

with torch.no_grad():
    for sample in dataloader:
        # Each sample is a tuple: (exp_x, exp_y, coeff, true_nodes_x, true_nodes_y, true_weights, id)
        exp_x, exp_y, coeff, true_values_x, true_values_y, true_values_w, id = sample

        # Move inputs to device and convert to float32.
        exp_x, exp_y, coeff = (
            exp_x.to(device, dtype=torch.float32, non_blocking=True),
            exp_y.to(device, dtype=torch.float32, non_blocking=True),
            coeff.to(device, dtype=torch.float32, non_blocking=True)
        )

        # Convert true values to numpy arrays (remain on CPU).
        true_nodes_x = true_values_x.numpy().astype(np.float32)
        true_nodes_y = true_values_y.numpy().astype(np.float32)
        true_weights = true_values_w.numpy().astype(np.float32)

        # Run inference on the model.
        predicted_values_x, predicted_values_y, predicted_values_w = model(exp_x, exp_y, coeff)

        # Process predictions: move to CPU and convert to numpy arrays.
        predicted_nodes_x = predicted_values_x.cpu().numpy().astype(np.float32)
        predicted_nodes_y = predicted_values_y.cpu().numpy().astype(np.float32)
        predicted_weights = predicted_values_w.cpu().numpy().astype(np.float32)

        # Compute integration values using the custom test function.
        pred_integral_tensor = utilities.compute_integration(predicted_values_x, predicted_values_y, predicted_values_w, test_fn)
        true_integral_tensor = utilities.compute_integration(true_values_x, true_values_y, true_values_w, test_fn)

        # Convert tensor outputs to float numbers.
        pred_val = pred_integral_tensor[0].item()
        true_val = true_integral_tensor[0].item()

        # Accumulate values for later visualization.
        predicted_integrals.append(pred_val)
        true_integrals.append(true_val)

        # Compute the absolute difference (for MAE).
        absolute_difference = abs(pred_val - true_val)
        total_absolute_difference += absolute_difference

        # --- START OF ADDITIONS FOR MSE ---
        squared_difference = (pred_val - true_val) ** 2
        total_squared_difference += squared_difference
        # --- END OF ADDITIONS FOR MSE ---

        total_ids += 1

        # Compute the relative error if the true value is nonzero.
        if abs(true_val) > 1e-10:
            rel_error = absolute_difference / abs(true_val)
        else:
            rel_error = 0.0  # Handle near-zero true values as needed.
        relative_errors.append(rel_error)
        rel_error_info.append((id[0], rel_error))

        # Print results to console (and to log file).
        print(f"Result of integration for {id}:")
        print(f"Algoim (True):  {true_val:.4e}")
        print(f"QuadNET (Pred): {pred_val:.4e}")
        print(f"Absolute Difference: {absolute_difference:.4e}")
        print(f"Relative Error: {rel_error*100:.2f}%")  # Multiply by 100 for percentage.

        # Plot the true and predicted nodes.
        plt.figure(figsize=(10, 6))

        # === NEW: reconstruct & draw the implicit boundary ===
        # build a grid over [-1,1]^2
        grid = np.linspace(-1, 1, 400)
        XX, YY = np.meshgrid(grid, grid)
        # convert tensors to 1D arrays
        exp_x_np = exp_x.cpu().numpy().reshape(-1)
        exp_y_np = exp_y.cpu().numpy().reshape(-1)
        coeff_np = coeff.cpu().numpy().reshape(-1)
        # evaluate f(x,y) = sum coeff_i * x^exp_x_i * y^exp_y_i
        ZZ = np.zeros_like(XX)
        for ex, ey, c in zip(exp_x_np, exp_y_np, coeff_np):
            ZZ += c * (XX**ex) * (YY**ey)
        # plot zero‐level contour
        plt.contour(XX, YY, ZZ, levels=[0], colors='k', linewidths=1.5)
        # ======================================================

        plt.scatter(true_nodes_x, true_nodes_y, c=true_weights, cmap='viridis',
                    label='Reference points (Algoim)', alpha=0.6, marker='x')
        plt.scatter(predicted_nodes_x, predicted_nodes_y, c=predicted_weights, cmap='plasma',
                    label='Predicted Points', alpha=0.6)
        plt.title('True vs Predicted Nodes')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.colorbar(label='Weight (Coefficient)')
        plt.legend()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        # Add an annotation in the top-left corner with integration values.
        plt.text(0.05, 0.95, f"True int (Algoim): {true_val:.8f}\nPred Int: {pred_val:.8f}",
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.5))
        sample_plot_path = os.path.join(output_folder, f'{id[0]}.png')
        plt.savefig(sample_plot_path)
        plt.close()

        # Save predictions to text file.
        with open(output_file, 'a') as f:
            f.write(
                f"{number};{id[0]};"  # Accessing the first element of the ID tuple.
                f"{','.join(map(str, predicted_nodes_x))};"
                f"{','.join(map(str, predicted_nodes_y))};"
                f"{','.join(map(str, predicted_weights))}\n"
            )

        number += 1

# After processing all samples, compute overall metrics.
overall_MAE = total_absolute_difference / total_ids if total_ids > 0 else 0
# --- START OF MSE COMPUTATION ---
overall_MSE = total_squared_difference / total_ids if total_ids > 0 else 0
# --- END OF MSE COMPUTATION ---
mean_relative_error = (sum(relative_errors) / total_ids * 100) if total_ids > 0 else 0
median_relative_error = (np.median(relative_errors) * 100) if total_ids > 0 else 0

# Print metrics to console (and log file).
print(f"Overall MAE: {overall_MAE:.4e}")
print(f"Overall MSE: {overall_MSE:.4e}")      # <--- New MSE output
print(f"Mean Relative Error: {mean_relative_error:.2f}%")
print(f"Median Relative Error: {median_relative_error:.2f}%")

# Identify outliers using the IQR method.
rel_errors_array = np.array(relative_errors)
Q1 = np.percentile(rel_errors_array, 25)
Q3 = np.percentile(rel_errors_array, 75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR

outlier_indices = np.where(rel_errors_array > upper_bound)[0]
print(f"Identified {len(outlier_indices)} outlier samples (relative error > {upper_bound*100:.2f}%):")
for idx in outlier_indices:
    sample_id, rel_err = rel_error_info[idx]
    print(f"Sample {sample_id}: Relative Error = {rel_err*100:.2f}%")

# Save overall metrics and outlier information to a subfolder.
metrics_folder = os.path.join(output_folder, "metrics")
os.makedirs(metrics_folder, exist_ok=True)
metrics_file = os.path.join(metrics_folder, "metrics.txt")
with open(metrics_file, 'w') as mf:
    mf.write(f"Overall MAE: {overall_MAE:.4e}\n")
    # --- ADD MSE TO METRICS FILE ---
    mf.write(f"Overall MSE: {overall_MSE:.4e}\n")
    # --- END ADDITION ---
    mf.write(f"Mean Relative Error: {mean_relative_error:.2f}%\n")
    mf.write(f"Median Relative Error: {median_relative_error:.2f}%\n")
    mf.write(f"Identified {len(outlier_indices)} outlier samples (relative error > {upper_bound*100:.2f}%):\n")
    for idx in outlier_indices:
        sample_id, rel_err = rel_error_info[idx]
        mf.write(f"Sample {sample_id}: Relative Error = {rel_err*100:.2f}%\n")

# Create a subfolder for outlier plots inside the output folder.
outliers_folder = os.path.join(output_folder, "outliers")
os.makedirs(outliers_folder, exist_ok=True)

# Copy outlier plots into the outliers folder.
for idx in outlier_indices:
    sample_id, _ = rel_error_info[idx]
    src_path = os.path.join(output_folder, f'{sample_id}.png')
    dst_path = os.path.join(outliers_folder, f'{sample_id}.png')
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"Copied plot for sample {sample_id} to outliers folder.")

# Visualization: Plot histogram of relative errors (in percentage) with mean annotations.
data = [x * 100 for x in relative_errors]

plt.figure(figsize=(10, 6))
# Create the histogram and capture the counts, bin edges, and patches.
counts, bins, patches = plt.hist(data, bins=20, edgecolor='black')

plt.xlabel("Relative Error (%)")
plt.ylabel("Frequency")
plt.title("Histogram of Relative Errors")

# Annotate each bar with the mean value of the data points in that bin.
for i in range(len(patches)):
    bin_lower = bins[i]
    bin_upper = bins[i + 1]
    # Select data points that fall into the current bin.
    bin_data = [d for d in data if d >= bin_lower and d < bin_upper]
    # Handle the last bin to include the upper edge.
    if i == len(patches) - 1:
        bin_data = [d for d in data if d >= bin_lower and d <= bin_upper]
    
    if bin_data:  # Only annotate if there is data in the bin.
        bin_mean = np.mean(bin_data)
        # Calculate the center of the current bin.
        bar_center = (bin_lower + bin_upper) / 2
        # Get the height (count) of the bar.
        bar_height = counts[i]
        # Annotate above the bar.
        plt.text(bar_center, bar_height, f"{bin_mean:.1f}", 
                 ha='center', va='bottom', fontsize=9)

# Save the histogram with annotations.
plt.savefig(os.path.join(output_folder, "relative_error_histogram.png"))
plt.close()

# Close the log file
log_file.close()
