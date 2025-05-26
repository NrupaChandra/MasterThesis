#!/usr/bin/env python

import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import os

# Swap in the FNN-based model instead of the CNN
from model_fnn import load_ff_pipelines_model
import utilities  # Must provide utilities.compute_integration(...)

###############################################################################
# Ellipse parameters
###############################################################################
a = 0.2          # first semi-axis
b = 0.35         # second semi-axis
C = (0.4, 0.6)   # center of the ellipse (C_x, C_y)
angle = np.pi/3  # rotation angle in radians (60°)

###############################################################################
# 1. Load the FNN Model
###############################################################################
device = torch.device('cpu')
model_path = r"C:\Git\MasterThesis\Models\FNN\FNN_model_v6\fnn_model_weights_v6.pth"
output_folder = r"C:\Git\MasterThesis\Scripts\FNN\FNN_V6\plt\ellipse"

# (1) construct the FNN with the same hyperparameters used in training
#     defaults: hidden_dim=2048, output_dim=1024, max_output_len=16, num_nodes=25,
#               domain=(-1,1), dropout_rate=0.0718234340555636, num_shared_layers=2
model = load_ff_pipelines_model(weights_path=None)

# (2) load the state dict onto CPU explicitly
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict)

# (3) move to device and switch to eval mode
model.to(device)
model.eval()

###############################################################################
# 2. Helper: Build the Ellipse Polynomial in a Subcell
###############################################################################
def make_subcell_ellipse_polynomial(ox, oy, n, device='cpu'):
    sub_half = 1.0 / n
    # Compute rotation and shift coefficients
    A_x = sub_half * np.cos(angle)
    A_y = sub_half * np.sin(angle)
    A_0 = (ox - C[0]) * np.cos(angle) + (oy - C[1]) * np.sin(angle)

    B_x = -sub_half * np.sin(angle)
    B_y = sub_half * np.cos(angle)
    B_0 = -(ox - C[0]) * np.sin(angle) + (oy - C[1]) * np.cos(angle)

    # Expand f_sub coefficients
    coeff_const = (A_0**2)/(a**2) + (B_0**2)/(b**2) - 1.0
    coeff_X     = (2*A_x*A_0)/(a**2) + (2*B_x*B_0)/(b**2)
    coeff_Y     = (2*A_y*A_0)/(a**2) + (2*B_y*B_0)/(b**2)
    coeff_X2    = (A_x**2)/(a**2) + (B_x**2)/(b**2)
    coeff_XY    = (2*A_x*A_y)/(a**2) + (2*B_x*B_y)/(b**2)
    coeff_Y2    = (A_y**2)/(a**2) + (B_y**2)/(b**2)

    exps_x = torch.tensor([[0,1,0,2,1,0]], dtype=torch.float32, device=device)
    exps_y = torch.tensor([[0,0,1,0,1,2]], dtype=torch.float32, device=device)
    coeffs = torch.tensor([[coeff_const, coeff_X, coeff_Y, coeff_X2, coeff_XY, coeff_Y2]], dtype=torch.float32, device=device)
    return exps_x, exps_y, coeffs

###############################################################################
# 3. Ellipse "Inside" Checker
###############################################################################
def is_inside_ellipse(x, y):
    x_shift = x - C[0]
    y_shift = y - C[1]
    Xp = x_shift * np.cos(angle) + y_shift * np.sin(angle)
    Yp = -x_shift * np.sin(angle) + y_shift * np.cos(angle)
    return (Xp**2)/(a**2) + (Yp**2)/(b**2) <= 1.0

###############################################################################
# 4. Subcell-based Integration with Full-Cell Check (using FNN)
###############################################################################
def compute_h_refined_integral_ellipse(n_subdivisions, model, device='cpu'):
    subcell_half = 1.0 / n_subdivisions
    jacobian = subcell_half**2
    centers = np.linspace(-1 + subcell_half, 1 - subcell_half, n_subdivisions)
    total_integral = 0.0

    for ox in centers:
        for oy in centers:
            if n_subdivisions == 1:
                exps_x_sub, exps_y_sub, coeffs_sub = make_subcell_ellipse_polynomial(ox, oy, n_subdivisions, device)
                with torch.no_grad():
                    px, py, pw = model(exps_x_sub, exps_y_sub, coeffs_sub)
                x_phys = ox + subcell_half * px
                y_phys = oy + subcell_half * py
                subcell_integral_tensor = utilities.compute_integration(
                    x_phys, y_phys, pw, lambda x, y: 1.0
                )
                subcell_integral = subcell_integral_tensor[0].item()
            else:
                # corner check
                corners_x = [ox - subcell_half, ox - subcell_half, ox + subcell_half, ox + subcell_half]
                corners_y = [oy - subcell_half, oy + subcell_half, oy - subcell_half, oy + subcell_half]
                inside_flags = [is_inside_ellipse(xc, yc) for xc, yc in zip(corners_x, corners_y)]
                center_flag = is_inside_ellipse(ox, oy)
                if all(inside_flags):
                    subcell_integral = 4.0
                elif not any(inside_flags) and not center_flag:
                    subcell_integral = 0.0
                else:
                    exps_x_sub, exps_y_sub, coeffs_sub = make_subcell_ellipse_polynomial(ox, oy, n_subdivisions, device)
                    with torch.no_grad():
                        px, py, pw = model(exps_x_sub, exps_y_sub, coeffs_sub)
                    x_phys = ox + subcell_half * px
                    y_phys = oy + subcell_half * py
                    subcell_integral_tensor = utilities.compute_integration(
                        x_phys, y_phys, pw, lambda x, y: 1.0
                    )
                    subcell_integral = subcell_integral_tensor[0].item()
            total_integral += jacobian * subcell_integral
    return total_integral

###############################################################################
# 5. Single Plot of All Subcells, Skipping FNN for Fully Inside
###############################################################################
def save_subcell_nodes_plot_ellipse(n_subdivisions, model, device='cpu', filename='subcell_nodes_ellipse.png'):
    subcell_half = 1.0 / n_subdivisions
    centers = np.linspace(-1 + subcell_half, 1 - subcell_half, n_subdivisions)

    partial_x_all, partial_y_all, partial_w_all = [], [], []
    plt.figure(figsize=(8,8))

    for ox in centers:
        for oy in centers:
            if n_subdivisions == 1:
                cell_case = 'partial'
            else:
                corners_x = [ox - subcell_half, ox - subcell_half, ox + subcell_half, ox + subcell_half]
                corners_y = [oy - subcell_half, oy + subcell_half, oy - subcell_half, oy + subcell_half]
                if all(is_inside_ellipse(xc, yc) for xc, yc in zip(corners_x, corners_y)):
                    cell_case = 'inside'
                elif not any(is_inside_ellipse(xc, yc) for xc, yc in zip(corners_x, corners_y)):
                    cell_case = 'outside'
                else:
                    cell_case = 'partial'

            if cell_case == 'inside':
                rect = plt.Rectangle((ox - subcell_half, oy - subcell_half), 2*subcell_half, 2*subcell_half,
                                     facecolor='lightgreen', edgecolor='blue', alpha=0.5, linestyle='--')
                plt.gca().add_patch(rect)
            elif cell_case == 'outside':
                rect = plt.Rectangle((ox - subcell_half, oy - subcell_half), 2*subcell_half, 2*subcell_half,
                                     facecolor='lightgray', edgecolor='blue', alpha=0.3, linestyle='--')
                plt.gca().add_patch(rect)
            else:
                exps_x_sub, exps_y_sub, coeffs_sub = make_subcell_ellipse_polynomial(ox, oy, n_subdivisions, device)
                with torch.no_grad():
                    px, py, pw = model(exps_x_sub, exps_y_sub, coeffs_sub)
                xs = (ox + subcell_half * px).cpu().numpy().ravel()
                ys = (oy + subcell_half * py).cpu().numpy().ravel()
                ws = pw.cpu().numpy().ravel()
                partial_x_all.append(xs)
                partial_y_all.append(ys)
                partial_w_all.append(ws)

    if partial_x_all:
        xs = np.concatenate(partial_x_all)
        ys = np.concatenate(partial_y_all)
        ws = np.concatenate(partial_w_all)
        sc = plt.scatter(xs, ys, c=ws, cmap='viridis', s=10, edgecolors='k')
        plt.colorbar(sc, label="Predicted Weight")

    # analytical ellipse boundary
    theta = np.linspace(0, 2*np.pi, 200)
    x_e = C[0] + a*np.cos(theta)*np.cos(angle) - b*np.sin(theta)*np.sin(angle)
    y_e = C[1] + a*np.cos(theta)*np.sin(angle) + b*np.sin(theta)*np.cos(angle)
    plt.plot(x_e, y_e, 'r-', linewidth=2, label='Ellipse Boundary')

    subcell_width = 2.0 / n_subdivisions
    for i in range(n_subdivisions+1):
        coord = -1 + i*subcell_width
        plt.axvline(x=coord, color='blue', linestyle='--', linewidth=0.5)
        plt.axhline(y=coord, color='blue', linestyle='--', linewidth=0.5)

    plt.title(f"Subcell-based Predicted Nodes for Ellipse (n={n_subdivisions} per dimension)")
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
# 6. Compute error_list and area_list for the Ellipse
###############################################################################
def compute_error_ellipse():
    analytical_area = math.pi * a * b
    refinement_levels = [1, 2, 4, 8,16,32]
    error_list, area_list = [], []

    print("\nComputing area by subdividing domain and calling FNN per subcell (with full-cell check) for the ellipse:")
    for n in refinement_levels:
        print(f"  Subcells: {n}x{n}")
        pred_area = compute_h_refined_integral_ellipse(n, model, device=device)
        area_list.append(pred_area)
        rel_error = abs(pred_area - analytical_area) / analytical_area
        error_list.append(rel_error)
        print(f"    Predicted area: {pred_area:.16f}")
        print(f"    Analytical area: {analytical_area:.16f}")
        print(f"    Relative error: {rel_error:.16f}\n")

        plot_fn = os.path.join(output_folder, f"predicted_nodes_ellipse_n{n}.png")
        save_subcell_nodes_plot_ellipse(n, model, device=device, filename=plot_fn)
        print(f"Aggregate subcell plot saved as '{plot_fn}'")

    # Plot Relative Error vs. Element Size
    element_sizes = [2.0 / n for n in refinement_levels]
    plt.figure(figsize=(8,6))
    plt.plot(element_sizes, error_list, marker='o', linestyle='-')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel("Element Size (2 / n) [log scale]")
    plt.ylabel("Relative Error [log scale]")
    plt.title("Relative Error vs. Element Size (Log-Log) for Ellipse with FNN")
    plt.grid(True, which='both', ls='--', alpha=0.7)
    err_fn = os.path.join(output_folder, "error_vs_element_size_ellipse_fnn.png")
    plt.savefig(err_fn, dpi=300)
    plt.close()
    print(f"Relative error vs. element size plot saved as '{err_fn}'")

    # Plot Integral Area vs. Refinement Level
    plt.figure(figsize=(8,6))
    plt.plot(refinement_levels, area_list, marker='o', linestyle='-')
    plt.axhline(y=analytical_area, color='r', linestyle='--', label='Analytical Area')
    plt.xlabel("Number of Subcells per Dimension (Refinement Level)")
    plt.ylabel("Integral Area")
    plt.title("Integral Area vs. Refinement Level for Ellipse with FNN")
    plt.legend(); plt.grid(True)
    area_fn = os.path.join(output_folder, "area_vs_refinement_ellipse_fnn.png")
    plt.savefig(area_fn, dpi=300)
    plt.close()
    print(f"Integral area plot saved as '{area_fn}'")

    return error_list, refinement_levels

###############################################################################
# 7. Main Script: Run everything if executed directly
###############################################################################
def main():
    compute_error_ellipse()

if __name__ == "__main__":
    main()
