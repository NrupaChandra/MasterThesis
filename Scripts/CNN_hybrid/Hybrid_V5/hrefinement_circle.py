#!/usr/bin/env python

import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import os

# Swap in the FNN-based model instead of the CNN
from model_hybrid import load_shallow_cnn_model
import utilities  # Must provide utilities.compute_integration(...)

###############################################################################
# 1. Load the FNN Model
###############################################################################
device = torch.device('cpu')
model_path = r"C:\Git\MasterThesis\Models\Hybrid\Hybrid_V5\fnn_model_weights_v6.pth"
output_folder = r"C:\Git\MasterThesis\Scripts\CNN_hybrid\Hybrid_V5\plt\circle"

# Construct the FNN and load its weights in one call
model = load_shallow_cnn_model(weights_path=model_path)
model.to(device)
model.eval()

###############################################################################
# 2. Helper: Build the Circle Polynomial in a Subcell (radius = 0.4)
###############################################################################
def make_subcell_circle_polynomial(ox, oy, n, device='cpu'):
    sub_half = 1.0 / n
    # f(x,y) = -1 + 6.25*(x^2+y^2)
    c_X2 = 6.25 * (sub_half**2)
    c_Y2 = 6.25 * (sub_half**2)
    c_X  = 12.5 * (ox * sub_half)
    c_Y  = 12.5 * (oy * sub_half)
    c_0  = -1.0 + 6.25*(ox**2 + oy**2)

    # Monomials: 1, X, X^2, Y, Y^2
    exps_x = torch.tensor([[0, 1, 2, 0, 0]], dtype=torch.float32, device=device)
    exps_y = torch.tensor([[0, 0, 0, 1, 2]], dtype=torch.float32, device=device)
    coeffs = torch.tensor([[c_0, c_X, c_X2, c_Y, c_Y2]], dtype=torch.float32, device=device)
    return exps_x, exps_y, coeffs

###############################################################################
# 3. Circle "Inside" Checker (radius = 0.4)
###############################################################################
def is_inside_circle(x, y):
    return (-1 + 6.25*x**2 + 6.25*y**2) <= 0

###############################################################################
# 4. Compute the integral over [−1,1]² by subdividing and using the FNN
###############################################################################
def compute_area_via_subdivision(n_subdivisions, full_cell_check=True):
    # cell centers and half-width
    grid = np.linspace(-1 + 1.0/n_subdivisions, 1 - 1.0/n_subdivisions, n_subdivisions)
    centers = grid.tolist()
    subcell_half = 1.0 / n_subdivisions

    areas = []
    for oy in centers:
        for ox in centers:
            # full cell inside?
            if full_cell_check and (
                is_inside_circle(ox - subcell_half, oy - subcell_half) and
                is_inside_circle(ox + subcell_half, oy - subcell_half) and
                is_inside_circle(ox - subcell_half, oy + subcell_half) and
                is_inside_circle(ox + subcell_half, oy + subcell_half)
            ):
                areas.append((2*subcell_half)**2)
                continue

            # otherwise evaluate via FNN on subcell
            exps_x_sub, exps_y_sub, coeffs_sub = make_subcell_circle_polynomial(ox, oy, n_subdivisions, device)
            with torch.no_grad():
                pred_nodes_x, pred_nodes_y, pred_weights = model(exps_x_sub, exps_y_sub, coeffs_sub)

            # map reference nodes → physical subcell coords
            x_phys = ox + subcell_half * pred_nodes_x
            y_phys = oy + subcell_half * pred_nodes_y

            # integrate on subcell
            subcell_integral_tensor = utilities.compute_integration(
                x_phys, y_phys, pred_weights, lambda x, y: 1.0
            )
            subcell_integral = subcell_integral_tensor[0].item()
            areas.append(subcell_integral)

    return sum(areas)

###############################################################################
# 5. Plotting Routine
###############################################################################
def plot_area_vs_refinement(refinement_levels, area_list, output_folder):
    plt.figure()
    plt.plot(refinement_levels, area_list, marker='o')
    plt.xscale('log', basex=2)
    plt.xlabel('Refinement Level (n subdivisions per side)')
    plt.ylabel('Computed Area')
    plt.title('Circle Area vs. Refinement Level (FNN)')
    plt.grid(True, which='both', ls='--', lw=0.5)
    filename = os.path.join(output_folder, "area_vs_refinement_fnn.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Area vs. refinement plot saved as '{filename}'")

###############################################################################
# 6. Compute error_list and area_list as a function to allow importing
###############################################################################
def compute_error_circle():
    analytical_area = math.pi * (0.4**2)
    refinement_levels = [1, 2, 4, 8, 16, 32, 64]
    error_list = []
    area_list = []

    print("\nComputing area by subdividing domain and calling FNN per subcell (with full-cell check):")
    for n in refinement_levels:
        area = compute_area_via_subdivision(n, full_cell_check=True)
        error = abs(area - analytical_area)
        print(f"  n={n:2d} → area={area:.6f}, error={error:.6e}")
        area_list.append(area)
        error_list.append(error)

    # Plot error vs. refinement
    plt.figure()
    plt.plot(refinement_levels, error_list, marker='o')
    plt.xscale('log', basex=2)
    plt.yscale('log')
    plt.xlabel('Refinement Level (n subdivisions per side)')
    plt.ylabel('Absolute Error')
    plt.title('Error vs. Refinement Level (FNN)')
    plt.grid(True, which='both', ls='--', lw=0.5)
    error_plot_fn = os.path.join(output_folder, "error_vs_refinement_fnn.png")
    plt.savefig(error_plot_fn, dpi=300)
    plt.close()
    print(f"Error vs. refinement plot saved as '{error_plot_fn}'")

    # Plot area vs. refinement
    plot_area_vs_refinement(refinement_levels, area_list, output_folder)

    # Also save area_list vs. error_list if needed
    area_plot_fn = os.path.join(output_folder, "area_vs_error_fnn.png")
    plt.figure()
    plt.plot(area_list, error_list, marker='o')
    plt.xlabel('Computed Area')
    plt.ylabel('Absolute Error')
    plt.title('Error vs. Computed Area (FNN)')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.savefig(area_plot_fn, dpi=300)
    plt.close()
    print(f"Integral area plot saved as '{area_plot_fn}'")

    return error_list, refinement_levels

###############################################################################
# 7. Main Script: Run everything if executed directly
###############################################################################
def main():
    compute_error_circle()

if __name__ == "__main__":
    main()
