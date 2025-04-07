#!/usr/bin/env python
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from model_cnn import load_shallow_cnn_model
import utilities  # Assumes utilities.compute_integration() is available

# Set device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model (adjust weights_path as needed).
model_path = "C:\\Git\\MasterThesis\\Models\\CNN\\CNN_V1\\cnn_model_weights_v1.0.pth"
model = load_shallow_cnn_model(weights_path=model_path, num_nodes=1225, domain=(-1,1), dropout_rate=0.0)

# Load the model with appropriate device mapping
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define the polynomial parameters for the circle function.
exp_x = torch.tensor([[0, 2, 0]], dtype=torch.float32, device=device)  # shape: (1, 3)
exp_y = torch.tensor([[0, 0, 2]], dtype=torch.float32, device=device)
coeff  = torch.tensor([[-1, 4, 4]], dtype=torch.float32, device=device)

with torch.no_grad():
    predicted_weights_tensor = model(exp_x, exp_y, coeff)  # shape: (batch, 1, grid_size, grid_size)
    batch_size = predicted_weights_tensor.size(0)
    num_nodes = model.nodal_preprocessor.num_nodes
    # Reshape the weight grid to (batch, num_nodes).
    predicted_weights_tensor = predicted_weights_tensor.view(batch_size, -1)
    # Retrieve the fixed nodal positions from the model's nodal preprocessor.
    predicted_nodes_x_tensor = model.nodal_preprocessor.X.unsqueeze(0).expand(batch_size, -1)
    predicted_nodes_y_tensor = model.nodal_preprocessor.Y.unsqueeze(0).expand(batch_size, -1)

# ---------------------------------------------------------------------------
# Define the test function (constant = 1) for integration over the domain.
# ---------------------------------------------------------------------------
def test_fn(x, y):
    return 1

# Compute the integral using your utilities function.
pred_integral_tensor = utilities.compute_integration(
    predicted_nodes_x_tensor, predicted_nodes_y_tensor, predicted_weights_tensor, test_fn
)
predicted_area = pred_integral_tensor[0].item()

# Convert from domain integral to circle area (4 - predicted_area).
predicted_area_circle = predicted_area

# Analytical area of a circle with radius 0.5.
analytical_area = math.pi * (0.5**2)

# Compute relative error (%) and mean squared error (MSE).
relative_error = abs(predicted_area_circle - analytical_area) / analytical_area * 100
mse = (predicted_area_circle - analytical_area) ** 2

print(f"Predicted area of circle: {predicted_area_circle:.16f}")
print(f"Analytical area of circle (r=0.5): {analytical_area:.16f}")
print(f"Relative Error: {relative_error:.6f}%")
print(f"MSE: {mse:.16e}")

# ---------------------------------------------------------------------------
# Visualization: Save a plot of the predicted nodal weights with the circle boundary.
# ---------------------------------------------------------------------------
predicted_nodes_x_np = predicted_nodes_x_tensor.cpu().numpy().flatten()
predicted_nodes_y_np = predicted_nodes_y_tensor.cpu().numpy().flatten()
predicted_weights_np = predicted_weights_tensor.cpu().numpy().flatten()

plt.figure(figsize=(7,7))

# Plot the nodes with small "x" markers and color by predicted weights.
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
theta = np.linspace(0, 2 * np.pi, 200)
x_circle = 0.5 * np.cos(theta)
y_circle = 0.5 * np.sin(theta)
plt.plot(x_circle, y_circle, 'r-', linewidth=2, label='Circle Boundary')

plt.title("Analytical solution - circle")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")

# Make the axes square so the circle isn't stretched.
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.grid(True)

# Annotate with area, relative error, and MSE.
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
plt.show()
plt.savefig("predicted_circle_integral.png", dpi=300)
plt.close()
