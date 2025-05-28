#!/usr/bin/env python

import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import os

from model_cnn import load_shallow_cnn_model
import utilities  # Must provide utilities.compute_integration(...)

###############################################################################
# 1. Load the CNN Model
###############################################################################
device = torch.device('cpu')
model_path = r"C:\Git\MasterThesis\Models\CNN\CNN_V6\cnn_model_weights_v5.0.pth"
output_folder = r"C:\Git\MasterThesis\Scripts\CNN\CNN_V6\plt\circle"

# (1) construct the model architecture without weights
model = load_shallow_cnn_model(None)

# (2) load the state dict onto CPU explicitly
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# (3) then move to your chosen device (in this case CPU)
model.to(device)
model.eval()


###############################################################################
# 2. Helper: Build the Circle Polynomial in a Subcell (radius = 0.4)
#    with reference‐cell normalization (divide by h^2)
###############################################################################
def make_subcell_circle_polynomial(ox, oy, n, device='cpu'):
    """
    Creates the normalized polynomial representation of the circle level-set:
      f_phys(x,y) = -1 + 6.25*(x^2+y^2)
    on the subcell center (ox, oy) of half-width h = 1/n,
    then rescales to reference coords [X,Y] in [-1,1] via
      f_ref(X,Y) = f_phys(ox + h*X, oy + h*Y) / h^2.

    Returns exps_x, exps_y, coeffs for
      f_ref(X,Y) = sum coeffs[i] * X^exps_x[i] * Y^exps_y[i]
    """
    h = 1.0 / n
    # Expand f_phys(ox + h X, oy + h Y) / h^2:
    # f_phys = -1 + 6.25*((ox + hX)^2 + (oy + hY)^2)
    # Divide each term by h^2
    c0 = (-1.0 + 6.25*(ox*ox + oy*oy)) / (h*h)
    cX = (12.5 * ox * h) / (h*h)    # = 12.5 * ox / h
    cY = (12.5 * oy * h) / (h*h)    # = 12.5 * oy / h
    cX2 = (6.25 * h*h) / (h*h)      # = 6.25
    cY2 = (6.25 * h*h) / (h*h)      # = 6.25

    exps_x = torch.tensor([[0, 1, 2, 0, 0]], dtype=torch.float32, device=device)
    exps_y = torch.tensor([[0, 0, 0, 1, 2]], dtype=torch.float32, device=device)
    coeffs = torch.tensor([[c0, cX, cX2, cY, cY2]], dtype=torch.float32, device=device)

    return exps_x, exps_y, coeffs

###############################################################################
# 3. Circle "Inside" Checker (radius = 0.4)
###############################################################################
def is_inside_circle(x, y):
    """
    Returns True if (x, y) is inside or on the circle defined by:
        f(x,y) = -1 + 6.25*(x^2+y^2) <= 0.
    """
    return (-1 + 6.25*x**2 + 6.25*y**2) <= 0

###############################################################################
# 4. Subcell-based Integration with Full-Cell Check + ML fallback
###############################################################################
def compute_h_refined_integral(n_subdivisions, model, device='cpu'):
    h = 1.0 / n_subdivisions
    jacobian = h*h
    centers = np.linspace(-1 + h, 1 - h, n_subdivisions)
    total_integral = 0.0

    nodes_x_ref = model.nodal_preprocessor.X.unsqueeze(0).to(device)
    nodes_y_ref = model.nodal_preprocessor.Y.unsqueeze(0).to(device)

    # threshold for ML fallback (single-precision eps ~1e-7)
    eps_fallback = 1e-4

    for ox in centers:
        for oy in centers:
            # Determine full/empty/partial
            if n_subdivisions > 1:
                corners = [(ox - h, oy - h), (ox - h, oy + h), (ox + h, oy - h), (ox + h, oy + h)]
                flags = [is_inside_circle(xc, yc) for xc, yc in corners]
                if all(flags):
                    subint = 4.0
                    total_integral += jacobian * subint
                    continue
                if not any(flags):
                    # outside
                    continue
            # partial (or n_subdivisions==1)
            # build normalized polynomial for this subcell
            exps_x_sub, exps_y_sub, coeffs_sub = make_subcell_circle_polynomial(ox, oy, n_subdivisions, device)
            with torch.no_grad():
                pred_weights = model(exps_x_sub, exps_y_sub, coeffs_sub).view(1, -1)
            # fallback: if all predicted weights tiny, treat as inside
            if torch.all(pred_weights.abs() < eps_fallback):
                subint = 4.0
            else:
                subint = utilities.compute_integration(
                    nodes_x_ref, nodes_y_ref, pred_weights, lambda x, y: 1.0
                )[0].item()
            total_integral += jacobian * subint

    return total_integral

###############################################################################
# 5. Single Plot of All Subcells, Skipping CNN for Fully Inside (modified for n==1)
###############################################################################
def save_subcell_nodes_plot(n_subdivisions, model, device='cpu', filename='subcell_nodes.png'):
    """
    Saves a plot of the subcell layout:
      - For n==1, the entire domain is treated as one (partial) cell and the CNN is called.
      - For n > 1, cells fully inside or outside are filled uniformly; partial cells are processed with the CNN.
    Overlays the analytical circle boundary and subcell grid lines.
    """
    subcell_half = 1.0 / n_subdivisions
    centers = np.linspace(-1 + subcell_half, 1 - subcell_half, n_subdivisions)

    partial_x_all = []
    partial_y_all = []
    partial_w_all = []

    nodes_x_ref = model.nodal_preprocessor.X.unsqueeze(0).to(device)
    nodes_y_ref = model.nodal_preprocessor.Y.unsqueeze(0).to(device)

    plt.figure(figsize=(8,8))

    for ox in centers:
        for oy in centers:
            if n_subdivisions == 1:
                cell_case = 'partial'
            else:
                corners_x = [ox - subcell_half, ox - subcell_half, ox + subcell_half, ox + subcell_half]
                corners_y = [oy - subcell_half, oy + subcell_half, oy - subcell_half, oy + subcell_half]
                if all(is_inside_circle(xc, yc) for xc, yc in zip(corners_x, corners_y)):
                    cell_case = 'inside'
                elif not any(is_inside_circle(xc, yc) for xc, yc in zip(corners_x, corners_y)):
                    cell_case = 'outside'
                else:
                    cell_case = 'partial'

            if cell_case == 'inside':
                rect = plt.Rectangle(
                    (ox - subcell_half, oy - subcell_half),
                    2*subcell_half, 2*subcell_half,
                    facecolor='lightgreen', edgecolor='blue', alpha=0.5, linestyle='--'
                )
                plt.gca().add_patch(rect)
            elif cell_case == 'outside':
                rect = plt.Rectangle(
                    (ox - subcell_half, oy - subcell_half),
                    2*subcell_half, 2*subcell_half,
                    facecolor='lightgray', edgecolor='blue', alpha=0.3, linestyle='--'
                )
                plt.gca().add_patch(rect)
            else:
                exps_x_sub, exps_y_sub, coeffs_sub = make_subcell_circle_polynomial(ox, oy, n_subdivisions, device)
                with torch.no_grad():
                    pred_weights = model(exps_x_sub, exps_y_sub, coeffs_sub).view(-1)
                x_mapped = subcell_half * nodes_x_ref + ox
                y_mapped = subcell_half * nodes_y_ref + oy

                partial_x_all.append(x_mapped.cpu().numpy().ravel())
                partial_y_all.append(y_mapped.cpu().numpy().ravel())
                partial_w_all.append(pred_weights.cpu().numpy().ravel())

    if partial_x_all:
        partial_x_all = np.concatenate(partial_x_all)
        partial_y_all = np.concatenate(partial_y_all)
        partial_w_all = np.concatenate(partial_w_all)
        sc = plt.scatter(
            partial_x_all, partial_y_all,
            c=partial_w_all, cmap='viridis',
            s=10, edgecolors='k'
        )
        plt.colorbar(sc, label="Predicted Weight")

    # Plot the analytical circle boundary with radius 0.4.
    theta = np.linspace(0, 2*np.pi, 200)
    x_circle = 0.4 * np.cos(theta)
    y_circle = 0.4 * np.sin(theta)
    plt.plot(x_circle, y_circle, 'r-', linewidth=2, label='Circle Boundary')

    subcell_width = 2.0 / n_subdivisions
    for i in range(n_subdivisions+1):
        coord = -1 + i * subcell_width
        plt.axvline(x=coord, color='blue', linestyle='--', linewidth=0.5)
        plt.axhline(y=coord, color='blue', linestyle='--', linewidth=0.5)

    plt.title(f"Subcell-based Predicted Nodes (n={n_subdivisions} per dimension)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

###############################################################################
# 6. Compute error_list and area_list as a function to allow importing
###############################################################################
def compute_error_circle():
    """
    Computes the predicted integral areas and relative errors.
    
    Returns:
        error_list (list): List of relative errors (fraction, not percentage).
        refinement_levels (list): List of refinement levels used.
    """
    analytical_area = math.pi * (0.4**2)  # Analytical area for a circle of radius 0.4
    refinement_levels = [1, 2, 4, 8,16]
    error_list = []  # Relative error for each refinement level.
    area_list = []   # Predicted integral areas.

    print("\nComputing area by subdividing domain and calling CNN per subcell (with full-cell check):")
    for n in refinement_levels:
        pred_area = compute_h_refined_integral(n, model, device=device)
        area_list.append(pred_area)
        rel_error = abs(pred_area - analytical_area) / analytical_area 
        error_list.append(rel_error)
        print(f"  Subcells: {n}x{n}")
        print(f"    Predicted area: {pred_area:.16f}")
        print(f"    Analytical area: {analytical_area:.16f}")
        print(f"    Relative error: {rel_error:.16f}\n")
        
        aggregated_plot_filename = os.path.join(output_folder, f"predicted_nodes_n{n}.png")
        save_subcell_nodes_plot(n, model, device=device, filename=aggregated_plot_filename)
        print(f"Aggregate subcell plot saved as '{aggregated_plot_filename}'.")

    # Plot Relative Error vs. Element Size (with log scale on both axes).
    element_sizes = [2.0 / n for n in refinement_levels]
    plt.figure(figsize=(8,6))
    plt.plot(element_sizes, error_list, marker='o', linestyle='-', color='b')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Element Size (2 / n) [log scale]")
    plt.ylabel("Relative Error [log scale]")
    plt.title("Relative Error vs. Element Size (Log-Log) v2")
    plt.grid(True, which='both', ls='--', alpha=0.7)
    plt.savefig(os.path.join(output_folder, "error_vs_element_size v2.png"), dpi=300)
    plt.close()
    print("Relative error vs. element size plot saved as 'error_vs_element_size v2.png'.")

    # Plot Integral Area vs. Refinement Level for reference.
    plt.figure(figsize=(8,6))
    plt.plot(refinement_levels, area_list, marker='o', linestyle='-', color='b', label='Predicted Integral Area')
    plt.axhline(y=analytical_area, color='r', linestyle='--', label='Analytical Area')
    plt.xlabel("Number of Subcells per Dimension (Refinement Level)")
    plt.ylabel("Integral Area")
    plt.title("Integral Area vs. Refinement Level")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "area_vs_refinement.png"), dpi=300)
    plt.close()
    print("Integral area plot saved as 'area_vs_refinement.png'.")

    return error_list, refinement_levels

###############################################################################
# 7. Main Script: Run everything if executed directly
###############################################################################
def main():
    compute_error_circle()

if __name__ == "__main__":
    main()
