#!/usr/bin/env python
import os
import shutil  # For copying files
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from model_gcnn import load_gnn_model, NodalPreprocessor
from multidataloader_fnn import MultiChunkDataset  
import utilities
import matplotlib.pyplot as plt
from torch_geometric.nn import knn_graph

# Set device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

# Define directories for model and test results.
model_dir   = "/work/scratch/ng66sume/Models/GCNN/GCNN_v3/"
results_dir = "/work/home/ng66sume/Test_Results/GCNN/GCNN_v3/"
os.makedirs(results_dir, exist_ok=True)

# Model hyperparams
in_channels     = 3           # [x, y, nodal_value]
hidden_channels = 64
num_layers      = 5
dropout_rate    = 0.0014281973712297197

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

# Load the model
model_path = os.path.join(model_dir, 'gcnn_model_weights_v3.pth')
model = load_gnn_model(in_channels, hidden_channels, num_layers, dropout_rate).to(device)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.eval()

# Test function for integration.
def test_fn(x, y):
    return 1

# Prepare dataset
data_dir = "/work/scratch/ng66sume/Root/Data/"
dataset = MultiChunkDataset(
    index_file=os.path.join(data_dir, 'combined_preprocessed_chunks_TestBernstein/index.txt'),
    base_dir=data_dir
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Output file
output_file = os.path.join(results_dir, "predicted_data_fnn.txt")
with open(output_file, 'w') as f:
    f.write("number;id;nodes_x;nodes_y;weights\n")

# Accumulators for metrics
predicted_integrals = []
true_integrals      = []
relative_errors     = []
rel_error_info      = []
squared_errors      = []
number              = 1
domain              = (-1.0, 1.0)
k_neighbors         = 4

# Preprocessor & graph connectivity
preprocessor = NodalPreprocessor(node_x_str, node_y_str).to(device)
pos          = torch.stack([preprocessor.X, preprocessor.Y], dim=1).to(device)
edge_index   = knn_graph(pos, k=k_neighbors, loop=False).to(device)

with torch.no_grad():
    for sample in dataloader:
        exp_x, exp_y, coeff, true_values_x, true_values_y, true_values_w, id = sample
        exp_x, exp_y, coeff = (
            exp_x.to(device, dtype=torch.float32),
            exp_y.to(device, dtype=torch.float32),
            coeff.to(device, dtype=torch.float32)
        )

        true_nodes_x = true_values_x.numpy().astype(np.float32)
        true_nodes_y = true_values_y.numpy().astype(np.float32)
        true_weights = true_values_w.numpy().astype(np.float32)

        raw = preprocessor(exp_x, exp_y, coeff)
        nodal_vals = raw.flatten().unsqueeze(1)
        node_feat  = torch.cat([pos, nodal_vals], dim=1)
        shifts, weights = model(node_feat, edge_index)

        pred_nodes         = pos + shifts
        predicted_values_x = pred_nodes[:, 0].unsqueeze(0)
        predicted_values_y = pred_nodes[:, 1].unsqueeze(0)
        predicted_values_w = weights.unsqueeze(0)

        predicted_nodes_x = predicted_values_x.cpu().numpy().astype(np.float32)
        predicted_nodes_y = predicted_values_y.cpu().numpy().astype(np.float32)
        predicted_weights = predicted_values_w.cpu().numpy().astype(np.float32)

        # Integrate
        pred_integral_tensor = utilities.compute_integration(predicted_values_x, predicted_values_y, predicted_values_w, test_fn)
        true_integral_tensor = utilities.compute_integration(true_values_x, true_values_y, true_values_w, test_fn)
        pred_val = pred_integral_tensor[0].item()
        true_val = true_integral_tensor[0].item()

        # Record for overall
        predicted_integrals.append(pred_val)
        true_integrals.append(true_val)

        # Absolute & relative errors
        absolute_difference = abs(pred_val - true_val)
        if abs(true_val) > 1e-10:
            rel_err = absolute_difference / abs(true_val)
        else:
            rel_err = 0.0
        relative_errors.append(rel_err)
        rel_error_info.append((id[0], rel_err))

        #  per‐sample MSE and RMSE 
        se     = absolute_difference**2
        squared_errors.append(se)
        

        # Prints
        print(f"Result for {id}:")
        print(f"  True Int:  {true_val:.4e}")
        print(f"  Pred Int:  {pred_val:.4e}")
        print(f"  Rel Error: {rel_err*100:.2f}%")
        print(f"  Sample MSE:  {se:.4e}")

        # Plotting (unchanged) …
        plt.figure(figsize=(10, 6))
        plt.scatter(true_nodes_x, true_nodes_y, c=true_weights, cmap='viridis',
                    label='True Points', alpha=0.6, marker='x')
        plt.scatter(predicted_nodes_x, predicted_nodes_y, c=predicted_weights, cmap='plasma',
                    label='Predicted Points', alpha=0.6)
        plt.title('True vs Predicted Nodes')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.colorbar(label='Weight')
        plt.legend()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.text(0.05, 0.95, f"True Int: {true_val:.8f}\nPred Int: {pred_val:.8f}",
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.5))
        sample_plot_path = os.path.join(results_dir, f'{id[0]}.png')
        plt.savefig(sample_plot_path)
        plt.close()

        # Append to output file
        with open(output_file, 'a') as f:
            f.write(
                f"{number};{id[0]};"
                f"{','.join(map(str, predicted_nodes_x))};"
                f"{','.join(map(str, predicted_nodes_y))};"
                f"{','.join(map(str, predicted_weights))}\n"
            )
        number += 1

# --- After loop: overall metrics ---
overall_MSE        = np.mean(squared_errors) if squared_errors else 0.0
mean_rel_error     = np.mean(relative_errors) * 100 if relative_errors else 0.0
median_rel_error   = np.median(relative_errors) * 100 if relative_errors else 0.0

print(f"Overall MSE:  {overall_MSE:.4e}")
print(f"Mean Relative Error:   {mean_rel_error:.2f}%")
print(f"Median Relative Error: {median_rel_error:.2f}%")

# Identify outliers (unchanged)
rel_errors_array = np.array(relative_errors)
Q1 = np.percentile(rel_errors_array, 25)
Q3 = np.percentile(rel_errors_array, 75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
outlier_indices = np.where(rel_errors_array > upper_bound)[0]

print(f"Identified {len(outlier_indices)} outlier samples (rel err > {upper_bound*100:.2f}%):")
for idx in outlier_indices:
    sample_id, rel_err = rel_error_info[idx]
    print(f"Sample {sample_id}: Rel Error = {rel_err*100:.2f}%")

# Save metrics
metrics_folder = os.path.join(results_dir, "metrics")
os.makedirs(metrics_folder, exist_ok=True)
metrics_file = os.path.join(metrics_folder, "metrics.txt")
with open(metrics_file, 'w') as mf:
    mf.write(f"Overall MSE:  {overall_MSE:.4e}\n")
    mf.write(f"Mean Rel Error:   {mean_rel_error:.2f}%\n")
    mf.write(f"Median Rel Error: {median_rel_error:.2f}%\n")
    mf.write(f"Identified {len(outlier_indices)} outlier samples (rel err > {upper_bound*100:.2f}%):\n")
    for idx in outlier_indices:
        sample_id, rel_err = rel_error_info[idx]
        mf.write(f"Sample {sample_id}: Rel Error = {rel_err*100:.2f}%\n")

# Copy outlier plots (unchanged) …
outliers_folder = os.path.join(results_dir, "outliers")
os.makedirs(outliers_folder, exist_ok=True)
for idx in outlier_indices:
    sample_id, _ = rel_error_info[idx]
    src = os.path.join(results_dir, f'{sample_id}.png')
    dst = os.path.join(outliers_folder, f'{sample_id}.png')
    if os.path.exists(src):
        shutil.copy(src, dst)

# Histogram of relative errors (unchanged) …
data = [x * 100 for x in relative_errors]
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(data, bins=20, edgecolor='black')
plt.xlabel("Relative Error (%)")
plt.ylabel("Frequency")
plt.title("Histogram of Relative Errors")
for i in range(len(patches)):
    bin_lower = bins[i]
    bin_upper = bins[i + 1]
    bin_data = [d for d in data if (d >= bin_lower and (d < bin_upper or i == len(patches)-1))]
    if bin_data:
        bar_center = 0.5*(bin_lower + bin_upper)
        plt.text(bar_center, counts[i], f"{np.mean(bin_data):.1f}",
                 ha='center', va='bottom', fontsize=9)
plt.savefig(os.path.join(results_dir, "relative_error_histogram.png"))
plt.close()
