#!/usr/bin/env python

import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import os

from model_fnn import load_ff_pipelines_model
import utilities 

###############################################################################
# Ellipse Parameters
###############################################################################
a = 0.2
b = 0.35
C = (0.4, 0.6)
angle = np.pi/3

###############################################################################
# Load the FNN Model
###############################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = r"C:\Git\MasterThesis\Models\FNN\FNN_model_v5.4\fnn_model_weights_v5.4.pth"
output_folder = r"C:\Git\MasterThesis\Scripts\FNN\FNN_V5.4\plt\ellipse"

model = load_ff_pipelines_model(weights_path=model_path)
model.to(device)
model.eval()

###############################################################################
# Polynomial Construction
###############################################################################
def make_subcell_ellipse_polynomial(ox, oy, n, device='cpu'):
    sub_half = 1.0 / n

    A_x = sub_half * np.cos(angle)
    A_y = sub_half * np.sin(angle)
    A_0 = (ox - C[0]) * np.cos(angle) + (oy - C[1]) * np.sin(angle)

    B_x = -sub_half * np.sin(angle)
    B_y = sub_half * np.cos(angle)
    B_0 = -(ox - C[0]) * np.sin(angle) + (oy - C[1]) * np.cos(angle)

    coeff_const = (A_0**2)/(a**2) + (B_0**2)/(b**2) - 1.0
    coeff_X     = (2*A_x*A_0)/(a**2) + (2*B_x*B_0)/(b**2)
    coeff_Y     = (2*A_y*A_0)/(a**2) + (2*B_y*B_0)/(b**2)
    coeff_X2    = (A_x**2)/(a**2) + (B_x**2)/(b**2)
    coeff_XY    = (2*A_x*A_y)/(a**2) + (2*B_x*B_y)/(b**2)
    coeff_Y2    = (A_y**2)/(a**2) + (B_y**2)/(b**2)

    exps_x = torch.tensor([[0, 1, 0, 2, 1, 0]], dtype=torch.float32, device=device)
    exps_y = torch.tensor([[0, 0, 1, 0, 1, 2]], dtype=torch.float32, device=device)
    coeffs = torch.tensor([[coeff_const, coeff_X, coeff_Y, coeff_X2, coeff_XY, coeff_Y2]], dtype=torch.float32, device=device)
    return exps_x, exps_y, coeffs

###############################################################################
# Inside Ellipse Checker
###############################################################################
def is_inside_ellipse(x, y):
    x_shift = x - C[0]
    y_shift = y - C[1]
    Xp = x_shift * np.cos(angle) + y_shift * np.sin(angle)
    Yp = -x_shift * np.sin(angle) + y_shift * np.cos(angle)
    return (Xp**2)/(a**2) + (Yp**2)/(b**2) <= 1.0

###############################################################################
# Integration Function
###############################################################################
def compute_h_refined_integral_ellipse(n_subdivisions, model, device='cpu'):
    subcell_half = 1.0 / n_subdivisions
    jacobian = subcell_half**2
    centers = np.linspace(-1 + subcell_half, 1 - subcell_half, n_subdivisions)
    total_integral = 0.0

    for ox in centers:
        for oy in centers:
            exps_x, exps_y, coeffs = make_subcell_ellipse_polynomial(ox, oy, n_subdivisions, device)
            if n_subdivisions == 1:
                call_model = True
            else:
                corners_x = [ox - subcell_half, ox - subcell_half, ox + subcell_half, ox + subcell_half]
                corners_y = [oy - subcell_half, oy + subcell_half, oy - subcell_half, oy + subcell_half]
                center_flag = is_inside_ellipse(ox, oy)
                inside_flags = [is_inside_ellipse(x, y) for x, y in zip(corners_x, corners_y)]
                if all(inside_flags):
                    total_integral += jacobian * 4.0
                    continue
                elif not any(inside_flags) and not center_flag:
                    continue
                call_model = True

            if call_model:
                with torch.no_grad():
                    x, y, w = model(exps_x, exps_y, coeffs)
                    x = x.view(1, -1)
                    y = y.view(1, -1)
                    w = w.view(1, -1)
                    result = utilities.compute_integration(x, y, w, lambda x, y: 1.0)
                    total_integral += jacobian * result[0].item()

    return total_integral

###############################################################################
# Visualization
###############################################################################
def save_subcell_nodes_plot_ellipse(n_subdivisions, model, device='cpu', filename='fnn_nodes_ellipse.png'):
    subcell_half = 1.0 / n_subdivisions
    centers = np.linspace(-1 + subcell_half, 1 - subcell_half, n_subdivisions)
    partial_x_all, partial_y_all, partial_w_all = [], [], []

    plt.figure(figsize=(8, 8))
    for ox in centers:
        for oy in centers:
            corners_x = [ox - subcell_half, ox - subcell_half, ox + subcell_half, ox + subcell_half]
            corners_y = [oy - subcell_half, oy + subcell_half, oy - subcell_half, oy + subcell_half]
            inside_flags = [is_inside_ellipse(x, y) for x, y in zip(corners_x, corners_y)]
            center_flag = is_inside_ellipse(ox, oy)

            if all(inside_flags):
                rect = plt.Rectangle((ox - subcell_half, oy - subcell_half), 2*subcell_half, 2*subcell_half,
                                     facecolor='lightgreen', edgecolor='blue', alpha=0.5, linestyle='--')
                plt.gca().add_patch(rect)
            elif not any(inside_flags) and not center_flag:
                rect = plt.Rectangle((ox - subcell_half, oy - subcell_half), 2*subcell_half, 2*subcell_half,
                                     facecolor='lightgray', edgecolor='blue', alpha=0.3, linestyle='--')
                plt.gca().add_patch(rect)
            else:
                exps_x, exps_y, coeffs = make_subcell_ellipse_polynomial(ox, oy, n_subdivisions, device)
                with torch.no_grad():
                    x, y, w = model(exps_x, exps_y, coeffs)
                x = x * subcell_half + ox
                y = y * subcell_half + oy
                partial_x_all.append(x.cpu().numpy().ravel())
                partial_y_all.append(y.cpu().numpy().ravel())
                partial_w_all.append(w.cpu().numpy().ravel())

    if partial_x_all:
        sc = plt.scatter(np.concatenate(partial_x_all), np.concatenate(partial_y_all),
                         c=np.concatenate(partial_w_all), cmap='viridis', s=10, edgecolors='k')
        plt.colorbar(sc, label='Predicted Weight')

    theta = np.linspace(0, 2*np.pi, 200)
    x_ellipse = C[0] + a * np.cos(theta) * np.cos(angle) - b * np.sin(theta) * np.sin(angle)
    y_ellipse = C[1] + a * np.cos(theta) * np.sin(angle) + b * np.sin(theta) * np.cos(angle)
    plt.plot(x_ellipse, y_ellipse, 'r-', linewidth=2, label='Ellipse Boundary')

    subcell_width = 2.0 / n_subdivisions
    for i in range(n_subdivisions+1):
        coord = -1 + i * subcell_width
        plt.axvline(x=coord, color='blue', linestyle='--', linewidth=0.5)
        plt.axhline(y=coord, color='blue', linestyle='--', linewidth=0.5)

    plt.title(f"Subcell-based Predicted Nodes for Ellipse (n={n_subdivisions})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

###############################################################################
# Error + Area Computation
###############################################################################
def compute_error_ellipse():
    analytical_area = math.pi * a * b
    refinement_levels = [1, 2, 4, 8, 16]
    error_list, area_list = [], []

    print("\nComputing FNN-based area predictions for Ellipse:")
    for n in refinement_levels:
        pred_area = compute_h_refined_integral_ellipse(n, model, device=device)
        area_list.append(pred_area)
        rel_error = abs(pred_area - analytical_area) / analytical_area
        error_list.append(rel_error)

        print(f"  n={n}: Area = {pred_area:.8f}, RelError = {rel_error:.2e}")
        save_subcell_nodes_plot_ellipse(n, model, device, filename=os.path.join(output_folder, f"fnn_nodes_ellipse_n{n}.png"))

    # Error vs Element Size
    element_sizes = [2.0 / n for n in refinement_levels]
    plt.figure()
    plt.loglog(element_sizes, error_list, 'o-b')
    plt.xlabel("Element Size (2/n)")
    plt.ylabel("Relative Error")
    plt.grid(True, which='both')
    plt.title("FNN: Error vs Element Size (Ellipse)")
    plt.savefig(os.path.join(output_folder, "fnn_error_vs_element_size_ellipse.png"), dpi=300)
    plt.close()

    # Area Plot
    plt.figure()
    plt.plot(refinement_levels, area_list, 'o-b', label='Predicted Area')
    plt.axhline(analytical_area, color='r', linestyle='--', label='Analytical Area')
    plt.xlabel("Refinement Level (n)")
    plt.ylabel("Integrated Area")
    plt.title("FNN: Area vs Refinement Level (Ellipse)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "fnn_area_vs_refinement_ellipse.png"), dpi=300)
    plt.close()

    return error_list, refinement_levels

###############################################################################
# Run
###############################################################################
def main():
    compute_error_ellipse()

if __name__ == "__main__":
    main()
