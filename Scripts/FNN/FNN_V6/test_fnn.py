#!/usr/bin/env python
import os
import shutil
import sys
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
device = torch.device('cpu')
torch.set_default_dtype(torch.float32)

# Define directories for model and test results.
model_dir = r"C:\Git\MasterThesis\Models\FNN\FNN_model_v6"
results_dir = r"C:\Git\RESULTS\Fnn"
os.makedirs(results_dir, exist_ok=True)

# Redirect all print() output (and errors) to a log file
log_path = os.path.join(results_dir, "console_output.txt")
sys.stdout = open(log_path, "w")
sys.stderr = sys.stdout

# Load the model and set to evaluation mode
model_path = os.path.join(model_dir, 'fnn_model_weights_v6.pth')
model = load_ff_pipelines_model(model_path).to(device)
model.eval()

def test_fn(x, y):
    return 1

# Data loading
data_dir = r"C:\Git\Data"
dataset = MultiChunkDataset(
    index_file=os.path.join(data_dir, 'combined_preprocessed_chunks_TestBernstein/index.txt'),
    base_dir=data_dir
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

output_folder = results_dir
os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(output_folder, "predicted_data_fnn.txt")
with open(output_file, 'w') as f:
    f.write("number;id;nodes_x;nodes_y;weights\n")

# Error tracking variables
total_absolute_difference = 0.0
total_squared_difference = 0.0
total_ids = 0
predicted_integrals = []
true_integrals = []
relative_errors = []
rel_error_info = []
number = 1

with torch.no_grad():
    for sample in dataloader:
        exp_x, exp_y, coeff, true_values_x, true_values_y, true_values_w, id = sample

        exp_x, exp_y, coeff = (exp_x.to(device, dtype=torch.float32),
                               exp_y.to(device, dtype=torch.float32),
                               coeff.to(device, dtype=torch.float32))

        true_nodes_x = true_values_x.numpy().astype(np.float32)
        true_nodes_y = true_values_y.numpy().astype(np.float32)
        true_weights = true_values_w.numpy().astype(np.float32)

        predicted_values_x, predicted_values_y, predicted_values_w = model(exp_x, exp_y, coeff)

        predicted_nodes_x = predicted_values_x.cpu().numpy().astype(np.float32)
        predicted_nodes_y = predicted_values_y.cpu().numpy().astype(np.float32)
        predicted_weights = predicted_values_w.cpu().numpy().astype(np.float32)

        pred_val = utilities.compute_integration(predicted_values_x, predicted_values_y, predicted_values_w, test_fn)[0].item()
        true_val = utilities.compute_integration(true_values_x, true_values_y, true_values_w, test_fn)[0].item()

        predicted_integrals.append(pred_val)
        true_integrals.append(true_val)

        absolute_difference = abs(pred_val - true_val)
        squared_difference = (pred_val - true_val) ** 2
        total_absolute_difference += absolute_difference
        total_squared_difference += squared_difference
        total_ids += 1

        rel_error = absolute_difference / abs(true_val) if abs(true_val) > 1e-10 else 0.0
        relative_errors.append(rel_error)
        rel_error_info.append((id[0], rel_error))

        print(f"Result of integration for {id}:")
        print(f"Algoim (True):  {true_val:.4e}")
        print(f"QuadNET (Pred): {pred_val:.4e}")
        print(f"Absolute Difference: {absolute_difference:.4e}")
        print(f"Relative Error: {rel_error*100:.2f}%")

        plt.figure(figsize=(10, 6))
        grid = np.linspace(-1, 1, 400)
        XX, YY = np.meshgrid(grid, grid)
        ZZ = np.zeros_like(XX)
        for ex, ey, c in zip(exp_x.cpu().numpy().reshape(-1), exp_y.cpu().numpy().reshape(-1), coeff.cpu().numpy().reshape(-1)):
            ZZ += c * (XX**ex) * (YY**ey)
        plt.contour(XX, YY, ZZ, levels=[0], colors='k', linewidths=1.5)
        plt.scatter(true_nodes_x, true_nodes_y, c=true_weights, cmap='viridis', label='Reference Points (Algoim)', alpha=0.6, marker='x')
        plt.scatter(predicted_nodes_x, predicted_nodes_y, c=predicted_weights, cmap='plasma', label='Predicted Points', alpha=0.6)
        plt.title('Reference(Algoim) vs Predicted Nodes')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.colorbar(label='Weight (Coefficient)')
        plt.legend()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.text(0.05, 0.95, f"True Int (Algoim): {true_val:.8f}\nPred Int : {pred_val:.8f}",
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.5))
        sample_plot_path = os.path.join(output_folder, f'{id[0]}.png')
        plt.savefig(sample_plot_path)
        plt.close()

        with open(output_file, 'a') as f:
            f.write(f"{number};{id[0]};{','.join(map(str, predicted_nodes_x))};{','.join(map(str, predicted_nodes_y))};{','.join(map(str, predicted_weights))}\n")

        number += 1

# Metrics
overall_MAE = total_absolute_difference / total_ids if total_ids > 0 else 0
overall_MSE = total_squared_difference / total_ids if total_ids > 0 else 0
mean_relative_error = (sum(relative_errors) / total_ids * 100) if total_ids > 0 else 0
median_relative_error = (np.median(relative_errors) * 100) if total_ids > 0 else 0

print(f"Overall MAE: {overall_MAE:.4e}")
print(f"Overall MSE: {overall_MSE:.4e}")
print(f"Mean Relative Error: {mean_relative_error:.2f}%")
print(f"Median Relative Error: {median_relative_error:.2f}%")

# Outlier detection
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

metrics_folder = os.path.join(output_folder, "metrics")
os.makedirs(metrics_folder, exist_ok=True)
metrics_file = os.path.join(metrics_folder, "metrics.txt")
with open(metrics_file, 'w') as mf:
    mf.write(f"Overall MAE: {overall_MAE:.4e}\n")
    mf.write(f"Overall MSE: {overall_MSE:.4e}\n")
    mf.write(f"Mean Relative Error: {mean_relative_error:.2f}%\n")
    mf.write(f"Median Relative Error: {median_relative_error:.2f}%\n")
    mf.write(f"Identified {len(outlier_indices)} outlier samples (relative error > {upper_bound*100:.2f}%):\n")
    for idx in outlier_indices:
        sample_id, rel_err = rel_error_info[idx]
        mf.write(f"Sample {sample_id}: Relative Error = {rel_err*100:.2f}%\n")

# Copy outlier plots
outliers_folder = os.path.join(output_folder, "outliers")
os.makedirs(outliers_folder, exist_ok=True)
for idx in outlier_indices:
    sample_id, _ = rel_error_info[idx]
    src_path = os.path.join(output_folder, f'{sample_id}.png')
    dst_path = os.path.join(outliers_folder, f'{sample_id}.png')
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"Copied plot for sample {sample_id} to outliers folder.")

# === Enhanced Visualization Block ===
data = [x * 100 for x in relative_errors]
plots_folder = os.path.join(output_folder, "plots")
os.makedirs(plots_folder, exist_ok=True)

# Plot 1: Annotated histogram
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(data, bins=20, edgecolor='black')
plt.xlabel("Relative Error (%)")
plt.ylabel("Frequency")
plt.title("Histogram of Relative Errors")
for i in range(len(patches)):
    bin_lower = bins[i]
    bin_upper = bins[i + 1]
    bin_data = [d for d in data if d >= bin_lower and d < bin_upper]
    if i == len(patches) - 1:
        bin_data = [d for d in data if d >= bin_lower and d <= bin_upper]
    if bin_data:
        bin_mean = np.mean(bin_data)
        bar_center = (bin_lower + bin_upper) / 2
        bar_height = counts[i]
        plt.text(bar_center, bar_height, f"{bin_mean:.1f}", ha='center', va='bottom', fontsize=9)
plt.axvline(x=mean_relative_error, color='red', linestyle='--', label='Mean')
plt.axvline(x=median_relative_error, color='green', linestyle='--', label='Median')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "relative_error_histogram.png"))
plt.close()

# Plot 2: Dual View - Zoom + Log Scale
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
axs[0].hist(data, bins=30, edgecolor='black', range=(0, 10))
axs[0].set_title("Zoomed-In Histogram (0–10%)")
axs[0].set_xlabel("Relative Error (%)")
axs[0].set_ylabel("Frequency")
axs[0].grid(True)
axs[1].hist(data, bins=np.logspace(-2, np.log10(max(data)+1), 50), edgecolor='black')
axs[1].set_xscale("log")
axs[1].set_title("Histogram (Log-Scaled X-Axis)")
axs[1].set_xlabel("Relative Error (%)")
axs[1].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "relative_error_dual_histogram.png"))
plt.close()
