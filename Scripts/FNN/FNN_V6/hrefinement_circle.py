#!/usr/bin/env python

import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import os

# Swap in the FNN-based model instead of the CNN
from model_fnn import load_ff_pipelines_model  # fileciteturn3file0
import utilities  # Must provide utilities.compute_integration(...)

###############################################################################
# 1. Load the FNN Model
###############################################################################
device = torch.device('cpu')
model_path = r"C:\Git\MasterThesis\Models\FNN\FNN_model_v6\fnn_model_weights_v6.pth"
output_folder = r"C:\Git\MasterThesis\Scripts\FNN\FNN_V6\plt\circle"

# (1) construct the FNN with the same hyperparameters used in training

model = load_ff_pipelines_model(weights_path=None)

# (2) load the state dict onto CPU explicitly
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict)

# (3) move to device and switch to eval mode
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
# 4. Subcell-based Integration with Full-Cell Check (using FNN)
###############################################################################
def compute_h_refined_integral(n_subdivisions, model, device='cpu'):
    subcell_half = 1.0 / n_subdivisions
    jacobian = subcell_half**2
    centers = np.linspace(-1 + subcell_half, 1 - subcell_half, n_subdivisions)
    total_integral = 0.0

    for ox in centers:
        for oy in centers:
            # always call FNN for n==1
            if n_subdivisions == 1:
                exps_x_sub, exps_y_sub, coeffs_sub = make_subcell_circle_polynomial(ox, oy, n_subdivisions, device)
                with torch.no_grad():
                    pred_nodes_x, pred_nodes_y, pred_weights = model(exps_x_sub, exps_y_sub, coeffs_sub)
                # map reference nodes → physical subcell coords
                x_phys = ox + subcell_half * pred_nodes_x
                y_phys = oy + subcell_half * pred_nodes_y
                subcell_integral_tensor = utilities.compute_integration(
                    x_phys, y_phys, pred_weights, lambda x, y: 1.0
                )
                subcell_integral = subcell_integral_tensor[0].item()

            else:
                # check corners
                corners_x = [ox - subcell_half, ox - subcell_half, ox + subcell_half, ox + subcell_half]
                corners_y = [oy - subcell_half, oy + subcell_half, oy - subcell_half, oy + subcell_half]
                inside_flags = [is_inside_circle(xc, yc) for xc, yc in zip(corners_x, corners_y)]
                if all(inside_flags):
                    subcell_integral = 4.0
                elif not any(inside_flags):
                    subcell_integral = 0.0
                else:
                    exps_x_sub, exps_y_sub, coeffs_sub = make_subcell_circle_polynomial(ox, oy, n_subdivisions, device)
                    with torch.no_grad():
                        pred_nodes_x, pred_nodes_y, pred_weights = model(exps_x_sub, exps_y_sub, coeffs_sub)
                    x_phys = ox + subcell_half * pred_nodes_x
                    y_phys = oy + subcell_half * pred_nodes_y
                    subcell_integral_tensor = utilities.compute_integration(
                        x_phys, y_phys, pred_weights, lambda x, y: 1.0
                    )
                    subcell_integral = subcell_integral_tensor[0].item()

            total_integral += jacobian * subcell_integral

    return total_integral

###############################################################################
# 5. Single Plot of All Subcells, Skipping FNN for Fully Inside
###############################################################################
def save_subcell_nodes_plot(n_subdivisions, model, device='cpu', filename='subcell_nodes.png'):
    subcell_half = 1.0 / n_subdivisions
    centers = np.linspace(-1 + subcell_half, 1 - subcell_half, n_subdivisions)

    partial_x_all = []
    partial_y_all = []
    partial_w_all = []

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
                    xn, yn, w = model(exps_x_sub, exps_y_sub, coeffs_sub)
                # physical coords
                x_mapped = (ox + subcell_half * xn).cpu().numpy().ravel()
                y_mapped = (oy + subcell_half * yn).cpu().numpy().ravel()
                w_mapped = w.cpu().numpy().ravel()
                partial_x_all.append(x_mapped)
                partial_y_all.append(y_mapped)
                partial_w_all.append(w_mapped)

    if partial_x_all:
        xs = np.concatenate(partial_x_all)
        ys = np.concatenate(partial_y_all)
        ws = np.concatenate(partial_w_all)
        sc = plt.scatter(xs, ys, c=ws, cmap='viridis', s=10, edgecolors='k')
        plt.colorbar(sc, label="Predicted Weight")

    # Plot analytical circle boundary
    theta = np.linspace(0, 2*np.pi, 200)
    x_circle = 0.4 * np.cos(theta)
    y_circle = 0.4 * np.sin(theta)
    plt.plot(x_circle, y_circle, 'r-', linewidth=2, label='Circle Boundary')

    # grid lines
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
    analytical_area = math.pi * (0.4**2)
    refinement_levels = [1, 2, 4, 8,16,32,64]
    error_list = []
    area_list = []

    print("\nComputing area by subdividing domain and calling FNN per subcell (with full-cell check):")
    for n in refinement_levels:
        print(f"  Subcells: {n}x{n}")
        pred_area = compute_h_refined_integral(n, model, device=device)
        area_list.append(pred_area)
        rel_error = abs(pred_area - analytical_area) / analytical_area
        error_list.append(rel_error)
        print(f"    Predicted area: {pred_area:.16f}")
        print(f"    Analytical area: {analytical_area:.16f}")
        print(f"    Relative error: {rel_error:.16f}\n")

        plot_fn = os.path.join(output_folder, f"predicted_nodes_n{n}.png")
        save_subcell_nodes_plot(n, model, device=device, filename=plot_fn)
        print(f"Aggregate subcell plot saved as '{plot_fn}'")

    # Plot Relative Error vs. Element Size (with log scale on both axes).
    element_sizes = [2.0 / n for n in refinement_levels]
    plt.figure(figsize=(8,6))
    plt.plot(element_sizes, error_list, marker='o', linestyle='-')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Element Size (2 / n) [log scale]")
    plt.ylabel("Relative Error [log scale]")
    plt.title("Relative Error vs. Element Size (Log-Log) with FNN")
    plt.grid(True, which='both', ls='--', alpha=0.7)
    err_plot_fn = os.path.join(output_folder, "error_vs_element_size_fnn.png")
    plt.savefig(err_plot_fn, dpi=300)
    plt.close()
    print(f"Relative error vs. element size plot saved as '{err_plot_fn}'")

    # Plot Integral Area vs. Refinement Level for reference.
    plt.figure(figsize=(8,6))
    plt.plot(refinement_levels, area_list, marker='o', linestyle='-')
    plt.axhline(y=analytical_area, color='r', linestyle='--', label='Analytical Area')
    plt.xlabel("Number of Subcells per Dimension (Refinement Level)")
    plt.ylabel("Integral Area")
    plt.title("Integral Area vs. Refinement Level with FNN")
    plt.legend()
    plt.grid(True)
    area_plot_fn = os.path.join(output_folder, "area_vs_refinement_fnn.png")
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