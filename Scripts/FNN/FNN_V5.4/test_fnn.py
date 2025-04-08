#!/usr/bin/env python
import os
import shutil  # For copying files
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from model_fnn import load_ff_pipelines_model
from multidataloader_fnn import MultiChunkDataset  
import utilities
import matplotlib.pyplot as plt

# Set device and default dtype (adjust as needed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

# Define directories for model and test results.
model_dir = "/work/home/ng66sume/MasterThesis/Models/FNN_model_v5.4/"
results_dir = "/work/home/ng66sume/MasterThesis/Test_Results/Test_FNN_v5.4/"
os.makedirs(results_dir, exist_ok=True)

# Load the model from the specified directory and set it to evaluation mode.
model_path = os.path.join(model_dir, 'fnn_model_weights_v5.4.pth')
model = load_ff_pipelines_model(model_path).to(device)
model.eval()

# Define a test function for integration.
def test_fn(x, y):
    return 1

# Define the data directory.
data_dir = "/work/scratch/ng66sume/Root/Data/"

# Create dataset using the base directory.
dataset = MultiChunkDataset(
    index_file=os.path.join(data_dir, 'preprocessed_chunks_TestBernstein/index.txt'),
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
        total_ids += 1

        # Compute the relative error if the true value is nonzero.
        if abs(true_val) > 1e-10:
            rel_error = absolute_difference / abs(true_val)
        else:
            rel_error = 0.0  # Handle near-zero true values as needed.
        relative_errors.append(rel_error)
        rel_error_info.append((id[0], rel_error))

        # Print results to console.
        print(f"Result of integration for {id}:")
        print(f"Algoim (True):  {true_val:.4e}")
        print(f"QuadNET (Pred): {pred_val:.4e}")
        print(f"Absolute Difference: {absolute_difference:.4e}")
        print(f"Relative Error: {rel_error*100:.2f}%")  # Multiply by 100 for percentage.

        # Plot the true and predicted nodes.
        plt.figure(figsize=(10, 6))
        plt.scatter(true_nodes_x, true_nodes_y, c=true_weights, cmap='viridis',
                    label='True Points', alpha=0.6, marker='x')
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
        plt.text(0.05, 0.95, f"True Int: {true_val:.8f}\nPred Int: {pred_val:.8f}",
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
mean_relative_error = (sum(relative_errors) / total_ids * 100) if total_ids > 0 else 0
median_relative_error = (np.median(relative_errors) * 100) if total_ids > 0 else 0

# Print metrics to console.
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

# Save overall metrics and outlier information to a subfolder.
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

'''# Visualization: Plot True vs. Predicted Integral Values
plt.figure(figsize=(8, 6))
plt.scatter(true_integrals, predicted_integrals, c='blue', label='Samples')
# Plot a reference line for ideal predictions (y = x).
min_val = min(min(true_integrals), min(predicted_integrals))
max_val = max(max(true_integrals), max(predicted_integrals))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal y=x')
plt.xlabel("True Integral")
plt.ylabel("Predicted Integral")
plt.title("True vs. Predicted Integral Values")
plt.legend()
# Add annotation for overall mean values in the upper left corner.
plt.text(0.05, 0.95, f"Mean True: {np.mean(true_integrals):.8f}\nMean Pred: {np.mean(predicted_integrals):.8f}",
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.5))
plt.savefig(os.path.join(output_folder, "true_vs_predicted_integrals.png"))
plt.close()'''
