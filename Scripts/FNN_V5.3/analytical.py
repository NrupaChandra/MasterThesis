#!/usr/bin/env python
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from model_fnn import load_ff_pipelines_model
import utilities  # Assumes utilities.compute_integration() is available

# Set device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model (adjust weights_path as needed).
model_path = "/work/scratch/ng66sume/Models/FNN_model_v5.3/fnn_model_weights_v4.2.pth"
# Ensure that the model architecture (num_nodes, etc.) matches the checkpoint.
model = load_ff_pipelines_model(model_path, num_nodes=25).to(device)
model.eval()

# Define the polynomial parameters for the circle function.
# Representing the circle with f(x,y) = -1 + 4*x^2 + 4*y^2 (an approximate representation)
exp_x = torch.tensor([[0, 2, 0]], dtype=torch.float32, device=device)  # shape: (1, 3)
exp_y = torch.tensor([[0, 0, 2]], dtype=torch.float32, device=device)
coeff  = torch.tensor([[-1, 4, 4]], dtype=torch.float32, device=device)

with torch.no_grad():
    # The model returns a tuple: (pred_nodes_x, pred_nodes_y, pred_weights)
    pred_nodes_x, pred_nodes_y, pred_weights = model(exp_x, exp_y, coeff)
    # Note: We use these torch tensors directly for integration.
    
    # Also convert to NumPy arrays later for plotting.
    predicted_nodes_x_np = pred_nodes_x.cpu().numpy().astype(np.float32).flatten()
    predicted_nodes_y_np = pred_nodes_y.cpu().numpy().astype(np.float32).flatten()
    predicted_weights_np = pred_weights.cpu().numpy().astype(np.float32).flatten()

# ---------------------------------------------------------------------------
# Define the test function (constant = 1) for integration over the domain.
# ---------------------------------------------------------------------------
def test_fn(x, y):
    return 1

# Compute the integral using your utilities function.
# Note: We're using torch tensors here.
pred_integral_tensor = utilities.compute_integration(pred_nodes_x, pred_nodes_y, pred_weights, test_fn)
predicted_area = pred_integral_tensor[0].item()

# In this example, we assume the predicted integral corresponds directly to the circle's area.
predicted_area_circle = predicted_area

# Analytical area of a circle with radius 0.5.
analytical_area = math.pi * (0.5**2)

# Compute relative error (%) and mean squared error (MSE).
relative_error = abs(predicted_area_circle - analytical_area) / analytical_area * 100
mse = (predicted_area_circle - analytical_area) ** 2

print(f"Predicted area of circle: {predicted_area_circle:.6f}")
print(f"Analytical area of circle (r=0.5): {analytical_area:.6f}")
print(f"Relative Error: {relative_error:.2f}%")
print(f"MSE: {mse:.4e}")

# ---------------------------------------------------------------------------
# Visualization: Save a plot of the predicted nodal weights with the circle boundary.
# ---------------------------------------------------------------------------
plt.figure(figsize=(7,7))

# Plot the nodes with small "x" markers and color them by predicted weights.
scatter = plt.scatter(
    predicted_nodes_x_np, 
    predicted_nodes_y_np, 
    c=predicted_weights_np,
    cmap='viridis', 
    edgecolor='k', 
    marker='x',
    s=20, 
    label='Predicted Nodes'
)

# Add a colorbar for the predicted weights.
plt.colorbar(scatter, label='Predicted Weight')

# Plot the circle boundary in red.
theta = np.linspace(0, 2*np.pi, 200)
x_circle = 0.5 * np.cos(theta)
y_circle = 0.5 * np.sin(theta)
plt.plot(x_circle, y_circle, 'r-', linewidth=2, label='Circle Boundary')

plt.title("Analytical solution - circle")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")

# Make axes square so the circle isn't stretched.
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.grid(True)

# Annotate with predicted area, true area, relative error, and MSE.
annotation_text = (
    f"Predicted Area: {predicted_area_circle:.4f}\n"
    f"True Area: {analytical_area:.4f}\n"
    f"Relative Error: {relative_error:.2f}%\n"
    f"MSE: {mse:.4e}"
)
plt.text(
    0.05, 0.95,
    annotation_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.5)
)

# Save the plot to a file.
plt.savefig("predicted_circle_integral.png", dpi=300)
plt.close()
