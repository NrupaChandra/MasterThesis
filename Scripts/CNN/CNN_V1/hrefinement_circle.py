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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = r"C:\Git\MasterThesis\Models\CNN\CNN_V1\cnn_model_weights_v1.0.pth"
output_folder = r"C:\Git\MasterThesis\Scripts\CNN\CNN_V1\plt\circle"

model = load_shallow_cnn_model(
    weights_path=model_path,
    num_nodes=1225,       
    domain=(-1, 1),      
    dropout_rate=0.0
)
model.to(device)
model.eval()

###############################################################################
# 2. Helper: Build the Circle Polynomial in a Subcell (radius = 0.4)
###############################################################################
def make_subcell_circle_polynomial(ox, oy, n, device='cpu'):
    """
    Creates the polynomial representation of the circle level-set:
      f(x,y) = -1 + 6.25*(x^2+y^2)
    but shifted & scaled into a subcell centered at (ox, oy) with half-width 1/n.
    
    Returns:
      exps_x, exps_y, coeffs (tensors on the given device)
    such that:
      f_sub(X,Y) = sum_i coeffs[i] * X^(exps_x[i]) * Y^(exps_y[i])
    for (X,Y) in the subcell's local reference coords.
    """
    sub_half = 1.0 / n  # half-width of the subcell

    # For a circle of radius 0.4, we use: f(x,y) = -1 + 6.25*(x^2+y^2)
    c_X2 = 6.25 * (sub_half**2)
    c_Y2 = 6.25 * (sub_half**2)
    c_X  = 12.5 * (ox * sub_half)
    c_Y  = 12.5 * (oy * sub_half)
    c_0  = -1.0 + 6.25*(ox**2 + oy**2)

    exps_x = torch.tensor([[0, 1, 2, 0, 0]], dtype=torch.float32, device=device)
    exps_y = torch.tensor([[0, 0, 0, 1, 2]], dtype=torch.float32, device=device)
    coeffs = torch.tensor([[c_0, c_X, c_X2, c_Y, c_Y2]], dtype=torch.float32, device=device)

    return exps_x, exps_y, coeffs

###############################################################################
# 3. Circle "Inside" Checker (radius = 0.4)
###############################################################################
def is_inside_circle(x, y):
    """
    Returns True if (x, y) is inside or on the circle defined by:
        f(x,y) = -1 + 6.25*(x^2+y^2) <= 0.
    This circle has a radius of 0.4.
    """
    return (-1 + 6.25*x**2 + 6.25*y**2) <= 0

###############################################################################
# 4. Subcell-based Integration with Full-Cell Check (modified for n==1)
###############################################################################
def compute_h_refined_integral(n_subdivisions, model, device='cpu'):
    """
    For each subcell:
      - For n > 1, if the subcell is entirely inside the circle, assign full area;
        if entirely outside, assign zero; otherwise, call the CNN.
      - For n == 1, always call the CNN so that the entire domain is used.
      Multiply the subcell's integration result by the Jacobian and sum over all subcells.
    """
    subcell_half = 1.0 / n_subdivisions   # half-width in physical units
    jacobian = subcell_half**2              # Jacobian from reference [-1,1]^2 to physical subcell.
    centers = np.linspace(-1 + subcell_half, 1 - subcell_half, n_subdivisions)
    total_integral = 0.0

    nodes_x_ref = model.nodal_preprocessor.X.unsqueeze(0).to(device)
    nodes_y_ref = model.nodal_preprocessor.Y.unsqueeze(0).to(device)

    for ox in centers:
        for oy in centers:
            if n_subdivisions == 1:
                exps_x_sub, exps_y_sub, coeffs_sub = make_subcell_circle_polynomial(ox, oy, n_subdivisions, device)
                with torch.no_grad():
                    pred_weights = model(exps_x_sub, exps_y_sub, coeffs_sub)
                    pred_weights = pred_weights.view(1, -1)
                subcell_integral_tensor = utilities.compute_integration(
                    nodes_x_ref, nodes_y_ref, pred_weights, lambda x, y: 1.0
                )
                subcell_integral = subcell_integral_tensor[0].item()
            else:
                corners_x = [ox - subcell_half, ox - subcell_half, ox + subcell_half, ox + subcell_half]
                corners_y = [oy - subcell_half, oy + subcell_half, oy - subcell_half, oy + subcell_half]
                inside_flags = [is_inside_circle(x, y) for x, y in zip(corners_x, corners_y)]
                if all(inside_flags):
                    subcell_integral = 4.0
                elif not any(inside_flags):
                    subcell_integral = 0.0
                else:
                    exps_x_sub, exps_y_sub, coeffs_sub = make_subcell_circle_polynomial(ox, oy, n_subdivisions, device)
                    with torch.no_grad():
                        pred_weights = model(exps_x_sub, exps_y_sub, coeffs_sub)
                        pred_weights = pred_weights.view(1, -1)
                    subcell_integral_tensor = utilities.compute_integration(
                        nodes_x_ref, nodes_y_ref, pred_weights, lambda x, y: 1.0
                    )
                    subcell_integral = subcell_integral_tensor[0].item()
            total_integral += jacobian * subcell_integral

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
    refinement_levels = [1, 2, 4, 8, 16]
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
