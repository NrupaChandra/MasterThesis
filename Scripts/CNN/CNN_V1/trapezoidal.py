#!/usr/bin/env python3
import os
import shutil  # For copying files
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model_cnn import load_shallow_cnn_model
from multidataloader_fnn import MultiChunkDataset  
import utilities

# -----------------------------
# Integration using the trapezoidal rule
# -----------------------------
def integrate_trapizoidal_rule(exp_x, exp_y, coeff, grid_points):
 
    xs = np.linspace(-1, 1, grid_points)
    ys = np.linspace(-1, 1, grid_points)
    X, Y = np.meshgrid(xs, ys)
    
    F = np.zeros_like(X) 
    for ex, ey, c in zip(exp_x, exp_y, coeff):
        F += c * (X ** ex) * (Y ** ey) # construct the Bernstein polynomial
    
    # Build indicator: 1 where f(x,y) < 0, 0 elsewhere.
    indicator = (F < 0).astype(float)
    
    # 2D trapezoidal integration
    area = np.trapz(np.trapz(indicator, xs, axis=1), ys)
    return area

# -----------------------------
# Device and dtype setup
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

# -----------------------------
# Directories, model loading, and dataset creation
# -----------------------------
model_dir = "/work/scratch/ng66sume/Models/CNN/CNN_V1/"
results_dir = "/work/scratch/ng66sume/Test_Results/CNN/CNN_V1/"
os.makedirs(results_dir, exist_ok=True)

model_path = os.path.join(model_dir, 'cnn_model_weights_v1.0.pth')
model = load_shallow_cnn_model(model_path).to(device)
model.eval()

# Test function for integration (constant function 1)
def test_fn(x, y):
    return 1

data_dir = "/work/scratch/ng66sume/Root/Data/"
dataset = MultiChunkDataset(
    index_file=os.path.join(data_dir, 'preprocessed_chunks_TestBernstein/index.txt'),
    base_dir=data_dir
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

output_folder = results_dir
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "predicted_data_cnn.txt")
with open(output_file, 'w') as f:
    f.write("number;id;nodes_x;nodes_y;weights\n")

# -----------------------------
# Accumulators for metrics
# -----------------------------
total_absolute_difference_pred = 0.0  # For ML predicted vs. true
total_absolute_difference_trap = 0.0  # For trapezoidal vs. true
total_ids = 0

predicted_integrals = []
true_integrals = []
trapezoidal_integrals = []

relative_errors_pred = []  # relative error: ML vs. true
relative_errors_trap = []  # relative error: trapezoidal vs. true

rel_error_info = []
number = 1

# -----------------------------
# Main Loop: Process each sample
# -----------------------------
with torch.no_grad():
    for sample in dataloader:
        # Unpack sample:
        # (exp_x, exp_y, coeff, true_nodes_x, true_nodes_y, true_weights, id)
        exp_x, exp_y, coeff, true_nodes_x, true_nodes_y, true_weights, sample_id = sample

        # Move polynomial data to device.
        exp_x, exp_y, coeff = (
            exp_x.to(device, dtype=torch.float32, non_blocking=True),
            exp_y.to(device, dtype=torch.float32, non_blocking=True),
            coeff.to(device, dtype=torch.float32, non_blocking=True)
        )

        # Convert true nodes/weights to numpy.
        true_nodes_x_np = true_nodes_x.numpy().astype(np.float32)
        true_nodes_y_np = true_nodes_y.numpy().astype(np.float32)
        true_weights_np = true_weights.numpy().astype(np.float32)

        # -----------------------------
        # ML Model Inference for predicted integral
        # -----------------------------
        predicted_weights_tensor = model(exp_x, exp_y, coeff)  # shape: (batch, 1, grid_size, grid_size)
        batch_size = predicted_weights_tensor.size(0)
        grid_size = int(np.sqrt(model.nodal_preprocessor.num_nodes))

        # Get predicted nodes (fixed by model's nodal preprocessor).
        predicted_nodes_x_tensor = model.nodal_preprocessor.X.unsqueeze(0).expand(batch_size, -1)
        predicted_nodes_y_tensor = model.nodal_preprocessor.Y.unsqueeze(0).expand(batch_size, -1)
        predicted_weights_tensor = predicted_weights_tensor.view(batch_size, -1)

        predicted_nodes_x = predicted_nodes_x_tensor.cpu().numpy().astype(np.float32)
        predicted_nodes_y = predicted_nodes_y_tensor.cpu().numpy().astype(np.float32)
        predicted_weights = predicted_weights_tensor.cpu().numpy().astype(np.float32)

        # Compute predicted integral.
        pred_integral_tensor = utilities.compute_integration(
            predicted_nodes_x_tensor, predicted_nodes_y_tensor, predicted_weights_tensor, test_fn
        )
        # Compute true integral.
        true_integral_tensor = utilities.compute_integration(
            torch.tensor(true_nodes_x_np), torch.tensor(true_nodes_y_np), torch.tensor(true_weights_np), test_fn
        )

        pred_val = pred_integral_tensor[0].item()
        true_val = true_integral_tensor[0].item()

        predicted_integrals.append(pred_val)
        true_integrals.append(true_val)

        # -----------------------------
        # Trapezoidal Integral using polynomial
        # -----------------------------
        # If polynomial data is in tensors, convert to lists.
        if isinstance(exp_x, torch.Tensor):
            exp_x_list = exp_x.squeeze().tolist()
        else:
            exp_x_list = exp_x
        if isinstance(exp_y, torch.Tensor):
            exp_y_list = exp_y.squeeze().tolist()
        else:
            exp_y_list = exp_y
        if isinstance(coeff, torch.Tensor):
            coeff_list = coeff.squeeze().tolist()
        else:
            coeff_list = coeff

        trap_val = integrate_trapizoidal_rule(exp_x_list, exp_y_list, coeff_list, grid_points=35)
        trapezoidal_integrals.append(trap_val)

        # -----------------------------
        # Compute differences and relative errors
        # -----------------------------
        abs_diff_pred = abs(pred_val - true_val)
        abs_diff_trap = abs(trap_val - true_val)

        total_absolute_difference_pred += abs_diff_pred
        total_absolute_difference_trap += abs_diff_trap
        total_ids += 1

        rel_error_pred = abs_diff_pred / abs(true_val) if abs(true_val) > 1e-10 else 0.0
        rel_error_trap = abs_diff_trap / abs(true_val) if abs(true_val) > 1e-10 else 0.0

        relative_errors_pred.append(rel_error_pred)
        relative_errors_trap.append(rel_error_trap)
        rel_error_info.append((sample_id[0], rel_error_pred, rel_error_trap))

        # -----------------------------
        # Print results to console.
        # -----------------------------
        print(f"Result for sample {sample_id}:")
        print(f"  True Integral:         {true_val:.4e}")
        print(f"  Predicted Integral:    {pred_val:.4e}")
        print(f"  Trapezoidal Integral:  {trap_val:.4e}")
        print(f"  Abs Diff (Pred-True):  {abs_diff_pred:.4e}   Relative Error: {rel_error_pred*100:.6f}%")
        print(f"  Abs Diff (Trap-True):  {abs_diff_trap:.4e}   Relative Error: {rel_error_trap*100:.6f}%\n")

        # -----------------------------
        # Plot true vs predicted nodes (for visualization)
        # -----------------------------
        plt.figure(figsize=(10, 6))
        max_val = max(true_weights_np.max(), predicted_weights[0].max())
        plt.scatter(true_nodes_x_np, true_nodes_y_np, c=true_weights_np, cmap='gray', vmin=0.0, vmax=max_val,
                    label='True Points', alpha=0.6, marker='x')
        pred_mask = (predicted_weights[0] != 0.0)
        plt.scatter(predicted_nodes_x[0][pred_mask], predicted_nodes_y[0][pred_mask],
                    c=predicted_weights[0][pred_mask], cmap='gray', vmin=0.0, vmax=max_val,
                    label='Predicted Points', alpha=0.6)
        plt.title('True vs. Predicted Nodes (Grayscale)')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.colorbar(label='Weight (Coefficient)')
        plt.legend()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.text(0.05, 0.95, f"True Int: {true_val:.8f}\nPred Int: {pred_val:.8f}\nTrap Int: {trap_val:.8f}",
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.5))
        sample_plot_path = os.path.join(output_folder, f'{sample_id[0]}.png')
        plt.savefig(sample_plot_path)
        plt.close()

        # Save predictions to file.
        with open(output_file, 'a') as f:
            f.write(f"{number};{sample_id[0]};"
                    f"{','.join(map(str, predicted_nodes_x[0]))};"
                    f"{','.join(map(str, predicted_nodes_y[0]))};"
                    f"{','.join(map(str, predicted_weights[0]))}\n")
        number += 1

# -----------------------------
# Compute overall metrics and report
# -----------------------------
overall_MAE_pred = total_absolute_difference_pred / total_ids if total_ids > 0 else 0.0
overall_MAE_trap = total_absolute_difference_trap / total_ids if total_ids > 0 else 0.0
mean_relative_error_pred = (sum(relative_errors_pred) / total_ids * 100) if total_ids > 0 else 0.0
mean_relative_error_trap = (sum(relative_errors_trap) / total_ids * 100) if total_ids > 0 else 0.0
median_relative_error_pred = (np.median(relative_errors_pred) * 100) if total_ids > 0 else 0.0
median_relative_error_trap = (np.median(relative_errors_trap) * 100) if total_ids > 0 else 0.0

print(f"Overall MAE (Predicted vs. True): {overall_MAE_pred:.4e}")
print(f"Overall MAE (Trapezoidal vs. True): {overall_MAE_trap:.4e}")
print(f"Mean Relative Error (Predicted vs. True): {mean_relative_error_pred:.6f}%")
print(f"Mean Relative Error (Trapezoidal vs. True): {mean_relative_error_trap:.6f}%")
#print(f"Median Relative Error (Predicted vs. True): {median_relative_error_pred:.6f}%")
#print(f"Median Relative Error (Trapezoidal vs. True): {median_relative_error_trap:.4f}%")

# Identify outliers using IQR (for predicted relative error).
rel_errors_array = np.array(relative_errors_pred)
Q1 = np.percentile(rel_errors_array, 25)
Q3 = np.percentile(rel_errors_array, 75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
outlier_indices = np.where(rel_errors_array > upper_bound)[0]
print(f"Identified {len(outlier_indices)} outlier samples (relative error (Predicted vs. True) > {upper_bound*100:.2f}%):")
for idx in outlier_indices:
    sample_id, rel_err_pred, rel_err_trap = rel_error_info[idx]
    print(f"Sample {sample_id}: Relative Error (Pred vs. True) = {rel_err_pred*100:.2f}%")

# Save metrics and outlier info.
metrics_folder = os.path.join(output_folder, "metrics")
os.makedirs(metrics_folder, exist_ok=True)
metrics_file = os.path.join(metrics_folder, "metrics.txt")
with open(metrics_file, 'w') as mf:
    mf.write(f"Overall MAE (Predicted vs. True): {overall_MAE_pred:.4e}\n")
    mf.write(f"Overall MAE (Trapezoidal vs. True): {overall_MAE_trap:.4e}\n")
    mf.write(f"Mean Relative Error (Predicted vs. True): {mean_relative_error_pred:.2f}%\n")
    mf.write(f"Mean Relative Error (Trapezoidal vs. True): {mean_relative_error_trap:.2f}%\n")
    mf.write(f"Median Relative Error (Predicted vs. True): {median_relative_error_pred:.2f}%\n")
    mf.write(f"Median Relative Error (Trapezoidal vs. True): {median_relative_error_trap:.2f}%\n")
    mf.write(f"Identified {len(outlier_indices)} outlier samples (relative error (Predicted vs. True) > {upper_bound*100:.2f}%):\n")
    for idx in outlier_indices:
        sample_id, rel_err_pred, _ = rel_error_info[idx]
        mf.write(f"Sample {sample_id}: Relative Error (Predicted vs. True) = {rel_err_pred*100:.2f}%\n")

# Copy outlier plots into a separate folder.
outliers_folder = os.path.join(output_folder, "outliers")
os.makedirs(outliers_folder, exist_ok=True)
for idx in outlier_indices:
    sample_id, _, _ = rel_error_info[idx]
    src_path = os.path.join(output_folder, f'{sample_id}.png')
    dst_path = os.path.join(outliers_folder, f'{sample_id}.png')
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"Copied plot for sample {sample_id} to outliers folder.")

# Plot histogram of relative errors (in percentage) for predicted vs. true.
plt.figure(figsize=(10, 6))
data = [x * 100 for x in relative_errors_pred]
counts, bins, patches = plt.hist(data, bins=20, edgecolor='black')
plt.xlabel("Relative Error (%)")
plt.ylabel("Frequency")
plt.title("Histogram of Relative Errors (Predicted vs. True)")
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
        plt.text(bar_center, bar_height, f"{bin_mean:.1f}", ha='center', va='bottom', fontsize=9)
plt.savefig(os.path.join(output_folder, "relative_error_histogram.png"))
plt.close()
