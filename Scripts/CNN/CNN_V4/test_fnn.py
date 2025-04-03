#!/usr/bin/env python
import os
import shutil  # For copying files
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
model_dir = "/work/scratch/ng66sume/Models/CNN/CNN_V4/"
results_dir = "/work/scratch/ng66sume/Test_Results/CNN/CNN_V4/"
os.makedirs(results_dir, exist_ok=True)

# Domain and fixed grid settings.
domain = (-1, 1)
pred_grid_size = 16  # fixed grid size for output

# Load the model checkpoint and set to evaluation mode.
# Note: the checkpoint here is assumed to be from your dual-branch model.
model_path = os.path.join(model_dir, 'cnn_model_weights_v3.0.pth')
model = load_shallow_cnn_model(model_path).to(device)
model.eval()

# Example test function for integration.
def test_fn(x, y):
    return 1  # constant function

# Define data directory.
data_dir = "/work/scratch/ng66sume/Root/Data/"

# Create dataset using the base directory.
dataset = MultiChunkDataset(
    index_file=os.path.join(data_dir, 'preprocessed_chunks_TestBernstein/index.txt'),
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

        # Model inference -> predicted weight grid and delta corrections.
        # The model now returns two outputs: weights and delta corrections.
        pred_weights_tensor, delta_out = model(exp_x, exp_y, coeff)  # shapes: (batch, 1, 16, 16) and (batch, 2, 16, 16)
        batch_size = pred_weights_tensor.size(0)

        # Create a fixed 16x16 base grid.
        xs_pred = torch.linspace(domain[0], domain[1], pred_grid_size, device=device)
        ys_pred = torch.linspace(domain[0], domain[1], pred_grid_size, device=device)
        X_pred, Y_pred = torch.meshgrid(xs_pred, ys_pred, indexing='ij')
        base_nodes_x = X_pred.flatten().unsqueeze(0).expand(batch_size, -1)  # (batch, 256)
        base_nodes_y = Y_pred.flatten().unsqueeze(0).expand(batch_size, -1)  # (batch, 256)
        
        # Reshape delta corrections: from (batch, 2, 16, 16) to (batch, 2, 256)
        delta = delta_out.view(batch_size, 2, -1)
        delta_x = delta[:, 0, :]  # (batch, 256)
        delta_y = delta[:, 1, :]  # (batch, 256)
        
        # Compute the corrected node positions.
        predicted_nodes_x_tensor = base_nodes_x + delta_x
        predicted_nodes_y_tensor = base_nodes_y + delta_y
        
        # Flatten predicted weights to shape (batch, 256).
        pred_weights_tensor = pred_weights_tensor.view(batch_size, -1)
        
        # Convert predictions to numpy for plotting.
        predicted_nodes_x = predicted_nodes_x_tensor.cpu().numpy().astype(np.float32)
        predicted_nodes_y = predicted_nodes_y_tensor.cpu().numpy().astype(np.float32)
        predicted_weights = pred_weights_tensor.cpu().numpy().astype(np.float32)
        
        # Compute predicted integral using utilities.compute_integration().
        pred_integral_tensor = utilities.compute_integration(
            predicted_nodes_x_tensor, predicted_nodes_y_tensor, pred_weights_tensor, test_fn
        )
        # Compute true integral.
        true_integral_tensor = utilities.compute_integration(
            torch.tensor(true_nodes_x_np), torch.tensor(true_nodes_y_np), torch.tensor(true_weights_np), test_fn
        )
        
        pred_val = pred_integral_tensor[0].item()
        true_val = true_integral_tensor[0].item()
        
        predicted_integrals.append(pred_val)
        true_integrals.append(true_val)
        
        # MAE and relative error calculations.
        absolute_difference = abs(pred_val - true_val)
        total_absolute_difference += absolute_difference
        total_ids += 1
        
        if abs(true_val) > 1e-10:
            rel_error = absolute_difference / abs(true_val)
        else:
            rel_error = 0.0
        relative_errors.append(rel_error)
        rel_error_info.append((id[0], rel_error))
        
        # Print results to console.
        print(f"Result of integration for {id}:")
        print(f"True Integral:         {true_val:.4e}")
        print(f"Predicted Integral:    {pred_val:.4e}")
        print(f"Absolute Difference:   {absolute_difference:.4e}")
        print(f"Relative Error:        {rel_error*100:.2f}%")
        
        # -----------------------------
        # Plot the true and predicted nodes in grayscale:
        #   - 0 weight = black
        #   - max weight = white
        #   - omit predicted nodes with 0 weight
        # -----------------------------
        plt.figure(figsize=(10, 6))
        
        # Determine maximum weight for consistent scaling in this sample.
        max_val = max(true_weights_np.max(), predicted_weights[0].max())
        
        # Plot true nodes (no filtering).
        plt.scatter(
            true_nodes_x_np, 
            true_nodes_y_np,
            c=true_weights_np,
            cmap='gray',
            vmin=0.0,
            vmax=max_val,
            label='True Points',
            alpha=0.6,
            marker='x'
        )
        
        # Create a mask to omit predicted nodes that have zero weight.
        pred_mask = (predicted_weights[0] != 0.0)
        
        plt.scatter(
            predicted_nodes_x[0][pred_mask],
            predicted_nodes_y[0][pred_mask],
            c=predicted_weights[0][pred_mask],
            cmap='gray',
            vmin=0.0,
            vmax=max_val,
            label='Predicted Points',
            alpha=0.6
        )
        
        plt.title('True vs. Predicted Nodes (Grayscale)')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        cb = plt.colorbar(label='Weight (Coefficient)')
        plt.legend()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        
        # Annotate with integration values.
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
overall_MAE = total_absolute_difference / total_ids if total_ids > 0 else 0
mean_relative_error = (sum(relative_errors) / total_ids * 100) if total_ids > 0 else 0
median_relative_error = (np.median(relative_errors) * 100) if total_ids > 0 else 0

print(f"Overall MAE: {overall_MAE:.4e}")
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

# Save metrics and outlier info.
metrics_folder = os.path.join(output_folder, "metrics")
os.makedirs(metrics_folder, exist_ok=True)
metrics_file = os.path.join(metrics_folder, "metrics.txt")
with open(metrics_file, 'w') as mf:
    mf.write(f"Overall MAE: {overall_MAE:.4e}\n")
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
    bin_data = [d for d in data if bin_lower <= d < bin_upper]
    if i == len(patches) - 1:
        bin_data = [d for d in data if d >= bin_lower and d <= bin_upper]
    if bin_data:
        bin_mean = np.mean(bin_data)
        bar_center = (bin_lower + bin_upper) / 2
        bar_height = counts[i]
        plt.text(bar_center, bar_height, f"{bin_mean:.1f}",
                 ha='center', va='bottom', fontsize=9)

plt.savefig(os.path.join(output_folder, "relative_error_histogram.png"))
plt.close()
