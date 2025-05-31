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
from model_cnn import load_shallow_cnn_model
from multidataloader_fnn import MultiChunkDataset  
import utilities
import matplotlib.pyplot as plt

# Set device and default dtype (adjust as needed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

# Define directories for model and test results.
model_dir = r"C:\Git\MasterThesis\Models\CNN\CNN_V6"
results_dir = r"C:\Git\RESULTS\Cnn"
os.makedirs(results_dir, exist_ok=True)

# Redirect all print() output (and errors) into a log file
log_path = os.path.join(results_dir, "console_output.txt")
sys.stdout = open(log_path, "w")
sys.stderr = sys.stdout

# Load the model checkpoint and set to evaluation mode.
model_path = os.path.join(model_dir, 'cnn_model_weights_v5.0.pth')
model = load_shallow_cnn_model(model_path).to(device)
model.eval()

# Example test function for integration.
def test_fn(x, y):
    return 1  # constant function

# Define data directory.
data_dir = r"C:\Git\Data"

# Create dataset using the base directory.
dataset = MultiChunkDataset(
    index_file=os.path.join(data_dir, 'combined_preprocessed_chunks_TestBernstein/index.txt'),
    base_dir=data_dir
)

# Use a batch size of 1 for testing.
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Create output folder for results.
output_folder = results_dir
os.makedirs(output_folder, exist_ok=True)

# File to save predictions.
output_file = os.path.join(output_folder, "predicted_data_cnn.txt")
with open(output_file, 'w') as f:
    f.write("number;id;nodes_x;nodes_y;weights\n")

# Accumulators for error metrics and integrals.
total_absolute_difference = 0.0
total_squared_error = 0.0       # <-- New accumulator for squared errors
total_ids = 0
predicted_integrals = []
true_integrals = []
relative_errors = []
rel_error_info = []
number = 1

with torch.no_grad():
    for sample in dataloader:
        # Each sample: (exp_x, exp_y, coeff, true_nodes_x, true_nodes_y, true_weights, id)
        exp_x, exp_y, coeff, true_nodes_x, true_nodes_y, true_weights, id = sample

        # Move inputs to device.
        exp_x, exp_y, coeff = (
            exp_x.to(device, dtype=torch.float32, non_blocking=True),
            exp_y.to(device, dtype=torch.float32, non_blocking=True),
            coeff.to(device, dtype=torch.float32, non_blocking=True)
        )

        # Convert true values to numpy (remain on CPU).
        true_nodes_x_np = true_nodes_x.numpy().astype(np.float32)
        true_nodes_y_np = true_nodes_y.numpy().astype(np.float32)
        true_weights_np = true_weights.numpy().astype(np.float32)

        # Model inference -> predicted weight grid.
        predicted_weights_tensor = model(exp_x, exp_y, coeff)
        batch_size = predicted_weights_tensor.size(0)

        # Obtain the fixed predicted nodes from the model's nodal preprocessor.
        predicted_nodes_x_tensor = model.nodal_preprocessor.X.unsqueeze(0).expand(batch_size, -1)
        predicted_nodes_y_tensor = model.nodal_preprocessor.Y.unsqueeze(0).expand(batch_size, -1)
        predicted_weights_tensor = predicted_weights_tensor.view(batch_size, -1)

        # Convert predictions to numpy for plotting.
        predicted_nodes_x = predicted_nodes_x_tensor.cpu().numpy().astype(np.float32)
        predicted_nodes_y = predicted_nodes_y_tensor.cpu().numpy().astype(np.float32)
        predicted_weights = predicted_weights_tensor.cpu().numpy().astype(np.float32)

        # Compute integrals.
        pred_integral_tensor = utilities.compute_integration(
            predicted_nodes_x_tensor, predicted_nodes_y_tensor, predicted_weights_tensor, test_fn
        )
        true_integral_tensor = utilities.compute_integration(
            torch.tensor(true_nodes_x_np), torch.tensor(true_nodes_y_np), torch.tensor(true_weights_np), test_fn
        )
        pred_val = pred_integral_tensor[0].item()
        true_val = true_integral_tensor[0].item()
        predicted_integrals.append(pred_val)
        true_integrals.append(true_val)

        # MAE, MSE, and relative error calculations.
        absolute_difference = abs(pred_val - true_val)
        squared_error = (pred_val - true_val) ** 2            # <-- compute squared error
        total_absolute_difference += absolute_difference
        total_squared_error += squared_error                  # <-- accumulate squared error
        total_ids += 1
        rel_error = absolute_difference / abs(true_val) if abs(true_val) > 1e-10 else 0.0
        relative_errors.append(rel_error)
        rel_error_info.append((id[0], rel_error))

        # Print results to console (now also into console_output.txt).
        print(f"Result of integration for {id}:")
        print(f"True Integral:         {true_val:.4e}")
        print(f"Predicted Integral:    {pred_val:.4e}")
        print(f"Absolute Difference:   {absolute_difference:.4e}")
        print(f"Relative Error:        {rel_error*100:.2f}%")

        # Plot the true and predicted nodes with colormaps.
        plt.figure(figsize=(10, 6))

        # Reconstruct & draw the implicit boundary
        grid = np.linspace(-1, 1, 400)
        XX, YY = np.meshgrid(grid, grid)
        exp_x_np = exp_x.cpu().numpy().reshape(-1)
        exp_y_np = exp_y.cpu().numpy().reshape(-1)
        coeff_np = coeff.cpu().numpy().reshape(-1)
        ZZ = np.zeros_like(XX)
        for ex, ey, c in zip(exp_x_np, exp_y_np, coeff_np):
            ZZ += c * (XX**ex) * (YY**ey)
        plt.contour(XX, YY, ZZ, levels=[0], colors='k', linewidths=1.5)

        # True points in viridis
        plt.scatter(
            true_nodes_x_np,
            true_nodes_y_np,
            c=true_weights_np,
            cmap='viridis',
            label='True Points',
            alpha=0.6,
            marker='x'
        )

        # Predicted points in plasma
        sc = plt.scatter(
            predicted_nodes_x[0],
            predicted_nodes_y[0],
            c=predicted_weights[0],
            cmap='plasma',
            label='Predicted Points',
            alpha=0.6
        )

        plt.title('True vs. Predicted Nodes')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.colorbar(sc, label='Weight (Coefficient)')
        plt.legend()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

        plt.text(
            0.05, 0.95,
            f"True Int: {true_val:.8f}\nPred Int: {pred_val:.8f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5)
        )

        sample_plot_path = os.path.join(output_folder, f'{id[0]}.png')
        plt.savefig(sample_plot_path)
        plt.close()

        # Save predictions to text file.
        with open(output_file, 'a') as f:
            f.write(
                f"{number};{id[0]};"
                f"{','.join(map(str, predicted_nodes_x[0]))};"
                f"{','.join(map(str, predicted_nodes_y[0]))};"
                f"{','.join(map(str, predicted_weights[0]))}\n"
            )

        number += 1

# Compute overall metrics.
overall_MAE = total_absolute_difference / total_ids if total_ids > 0 else 0.0
overall_MSE = total_squared_error / total_ids if total_ids > 0 else 0.0    # <-- compute MSE
mean_relative_error = (sum(relative_errors) / total_ids * 100) if total_ids > 0 else 0.0
median_relative_error = (np.median(relative_errors) * 100) if total_ids > 0 else 0.0

print(f"Overall MAE: {overall_MAE:.4e}")
print(f"Overall MSE: {overall_MSE:.4e}")                     # <-- print MSE
print(f"Mean Relative Error: {mean_relative_error:.2f}%")
print(f"Median Relative Error: {median_relative_error:.2f}%")

# Identify outliers using the IQR method.
rel_errors_array = np.array(relative_errors)
Q1, Q3 = np.percentile(rel_errors_array, [25, 75])
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR

outlier_indices = np.where(rel_errors_array > upper_bound)[0]
print(f"Identified {len(outlier_indices)} outlier samples (relative error > {upper_bound*100:.2f}%):")
for idx in outlier_indices:
    sample_id, rel_err = rel_error_info[idx]
    print(f"Sample {sample_id}: Relative Error = {rel_err*100:.2f}%")

# Save metrics and outlier info.
metrics_folder = os.path.join(output_folder, "metrics")
os.makedirs(metrics_folder, exist_ok=True)
metrics_file = os.path.join(metrics_folder, "metrics.txt")
with open(metrics_file, 'w') as mf:
    mf.write(f"Overall MAE: {overall_MAE:.4e}\n")
    mf.write(f"Overall MSE: {overall_MSE:.4e}\n")                   # <-- write MSE
    mf.write(f"Mean Relative Error: {mean_relative_error:.2f}%\n")
    mf.write(f"Median Relative Error: {median_relative_error:.2f}%\n")
    mf.write(f"Identified {len(outlier_indices)} outlier samples (relative error > {upper_bound*100:.2f}%):\n")
    for idx in outlier_indices:
        sample_id, rel_err = rel_error_info[idx]
        mf.write(f"Sample {sample_id}: Relative Error = {rel_err*100:.2f}%\n")

# Copy outlier plots into a separate folder.
outliers_folder = os.path.join(output_folder, "outliers")
os.makedirs(outliers_folder, exist_ok=True)
for idx in outlier_indices:
    sample_id, _ = rel_error_info[idx]
    src_path = os.path.join(output_folder, f'{sample_id}.png')
    dst_path = os.path.join(outliers_folder, f'{sample_id}.png')
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"Copied plot for sample {sample_id} to outliers folder.")

# Plot histogram of relative errors (in percentage).
data = [x * 100 for x in relative_errors]
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(data, bins=20, edgecolor='black')
plt.xlabel("Relative Error (%)")
plt.ylabel("Frequency")
plt.title("Histogram of Relative Errors")

for i in range(len(patches)):
    bin_lower = bins[i]
    bin_upper = bins[i + 1]
    bin_data = [d for d in data if bin_lower <= d < bin_upper] + \
               ([d for d in data if d == bin_upper] if i == len(patches) - 1 else [])
    if bin_data:
        bin_mean = np.mean(bin_data)
        bar_center = (bin_lower + bin_upper) / 2
        bar_height = counts[i]
        plt.text(bar_center, bar_height, f"{bin_mean:.1f}",
                 ha='center', va='bottom', fontsize=9)

plt.savefig(os.path.join(output_folder, "relative_error_histogram.png"))
plt.close()
