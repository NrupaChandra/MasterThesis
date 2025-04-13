#!/usr/bin/env python

import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import os

from model_cnn import load_shallow_cnn_model
import utilities  # Must provide utilities.compute_integration(...)

###############################################################################
# Ellipse parameters
###############################################################################
a = 0.2          # first semi-axis
b = 0.35         # second semi-axis
C = (0.4, 0.6)   # center of the ellipse (C_x, C_y)
angle = np.pi/3  # rotation angle in radians (60°)

###############################################################################
# 1. Load the CNN Model
###############################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = r"C:\Git\MasterThesis\Models\CNN\CNN_V1\cnn_model_weights_v1.0.pth"
output_folder = r"C:\Git\MasterThesis\Scripts\CNN\CNN_V1\plt\ellipse"

model = load_shallow_cnn_model(
    weights_path=model_path,
    num_nodes=1225,       
    domain=(-1, 1),      
    dropout_rate=0.0
)
model.to(device)
model.eval()

###############################################################################
# 2. Helper: Build the Ellipse Polynomial in a Subcell
###############################################################################
def make_subcell_ellipse_polynomial(ox, oy, n, device='cpu'):
    """
    Creates the polynomial representation of the ellipse level-set:
      f(x,y) = (X'^2)/(a^2) + (Y'^2)/(b^2) - 1,
    where for a point (x,y):
      X' = (x - C[0])*cos(angle) + (y - C[1])*sin(angle)
      Y' = -(x - C[0])*sin(angle) + (y - C[1])*cos(angle)
    and (x,y) is given in terms of local subcell coordinates (X,Y) by:
      x = sub_half * X + ox,  y = sub_half * Y + oy,
    with sub_half = 1/n.
    
    We expand this in powers of X and Y and represent it as:
      f_sub(X,Y) = sum coeffs[i] * X^(exps_x[i]) * Y^(exps_y[i])
    with monomials: constant, X, Y, X^2, X*Y, Y^2.
    """
    sub_half = 1.0 / n  # half-width of the subcell

    # The physical coordinates in the subcell:
    # x = sub_half * X + ox,   y = sub_half * Y + oy.
    # Define shifted coordinates:
    # (x - C[0]) = sub_half*X + (ox - C[0])
    # (y - C[1]) = sub_half*Y + (oy - C[1])
    #
    # Now apply the standard rotation:
    # X' = cos(angle)*(x - C[0]) + sin(angle)*(y - C[1])
    #    = sub_half*cos(angle)*X + sub_half*sin(angle)*Y + (ox - C[0])*cos(angle) + (oy - C[1])*sin(angle)
    # Y' = -sin(angle)*(x - C[0]) + cos(angle)*(y - C[1])
    #    = -sub_half*sin(angle)*X + sub_half*cos(angle)*Y - (ox - C[0])*sin(angle) + (oy - C[1])*cos(angle)
    #
    # Define coefficients for the linear forms:
    A_x = sub_half * np.cos(angle)
    A_y = sub_half * np.sin(angle)
    A_0 = (ox - C[0]) * np.cos(angle) + (oy - C[1]) * np.sin(angle)

    B_x = -sub_half * np.sin(angle)
    B_y = sub_half * np.cos(angle)
    B_0 = -(ox - C[0]) * np.sin(angle) + (oy - C[1]) * np.cos(angle)
    
    # Expand X'^2 and Y'^2:
    # X'^2 = A_x^2 X^2 + 2 A_x A_y X Y + A_y^2 Y^2 + 2 A_x A_0 X + 2 A_y A_0 Y + A_0^2
    # Y'^2 = B_x^2 X^2 + 2 B_x B_y X Y + B_y^2 Y^2 + 2 B_x B_0 X + 2 B_y B_0 Y + B_0^2
    #
    # f_sub(X,Y) = (X'^2)/(a^2) + (Y'^2)/(b^2) - 1.
    coeff_const = (A_0**2)/(a**2) + (B_0**2)/(b**2) - 1.0
    coeff_X     = (2*A_x*A_0)/(a**2) + (2*B_x*B_0)/(b**2)
    coeff_Y     = (2*A_y*A_0)/(a**2) + (2*B_y*B_0)/(b**2)
    coeff_X2    = (A_x**2)/(a**2) + (B_x**2)/(b**2)
    coeff_XY    = (2*A_x*A_y)/(a**2) + (2*B_x*B_y)/(b**2)
    coeff_Y2    = (A_y**2)/(a**2) + (B_y**2)/(b**2)

    # Represent the polynomial using the monomials: 1, X, Y, X^2, X*Y, Y^2.
    exps_x = torch.tensor([[0, 1, 0, 2, 1, 0]], dtype=torch.float32, device=device)
    exps_y = torch.tensor([[0, 0, 1, 0, 1, 2]], dtype=torch.float32, device=device)
    coeffs = torch.tensor([[coeff_const, coeff_X, coeff_Y, coeff_X2, coeff_XY, coeff_Y2]], dtype=torch.float32, device=device)

    return exps_x, exps_y, coeffs

###############################################################################
# 3. Ellipse "Inside" Checker (using standard rotation)
###############################################################################
def is_inside_ellipse(x, y):
    """
    Returns True if the point (x, y) is inside or on the ellipse defined by:
      (X'^2)/(a^2) + (Y'^2)/(b^2) - 1 <= 0,
    """
    x_shift = x - C[0]
    y_shift = y - C[1]
    Xp = x_shift * np.cos(angle) + y_shift * np.sin(angle)
    Yp = -x_shift * np.sin(angle) + y_shift * np.cos(angle)
    return (Xp**2)/(a**2) + (Yp**2)/(b**2) <= 1.0

###############################################################################
# 4. Subcell-based Integration with Full-Cell Check (for the Ellipse)
###############################################################################
def compute_h_refined_integral_ellipse(n_subdivisions, model, device='cpu'):
    """
    For each subcell:
      - For n > 1, if the subcell is entirely inside the ellipse, assign full area;
        if entirely outside, assign zero; otherwise, call the CNN.
      - For n == 1, always call the CNN so that the entire domain is used.
      Multiply the subcell's integration result by the Jacobian and sum over all subcells.
    """
    subcell_half = 1.0 / n_subdivisions  # half-width in physical units
    jacobian = subcell_half**2             # Jacobian factor
    centers = np.linspace(-1 + subcell_half, 1 - subcell_half, n_subdivisions)
    total_integral = 0.0

    nodes_x_ref = model.nodal_preprocessor.X.unsqueeze(0).to(device)
    nodes_y_ref = model.nodal_preprocessor.Y.unsqueeze(0).to(device)

    for ox in centers:
        for oy in centers:
            if n_subdivisions == 1:
                exps_x_sub, exps_y_sub, coeffs_sub = make_subcell_ellipse_polynomial(ox, oy, n_subdivisions, device)
                with torch.no_grad():
                    pred_weights = model(exps_x_sub, exps_y_sub, coeffs_sub)
                    pred_weights = pred_weights.view(1, -1)
                subcell_integral_tensor = utilities.compute_integration(
                    nodes_x_ref, nodes_y_ref, pred_weights, lambda x, y: 1.0
                )
                subcell_integral = subcell_integral_tensor[0].item()
            else:
                # Determine the four corners of the subcell.
                corners_x = [ox - subcell_half, ox - subcell_half, ox + subcell_half, ox + subcell_half]
                corners_y = [oy - subcell_half, oy + subcell_half, oy - subcell_half, oy + subcell_half]
                inside_flags = [is_inside_ellipse(x, y) for x, y in zip(corners_x, corners_y)]
                # Also checks the center of the subcell
                center_flag = [is_inside_ellipse(ox, oy)] 
                if all(inside_flags):
                    subcell_integral = 4.0
                elif not any(inside_flags) and not center_flag:
                    subcell_integral = 0.0
                else:
                    exps_x_sub, exps_y_sub, coeffs_sub = make_subcell_ellipse_polynomial(ox, oy, n_subdivisions, device)
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
# 5. Single Plot of All Subcells, Skipping CNN for Fully Inside (Ellipse Version)
###############################################################################
def save_subcell_nodes_plot_ellipse(n_subdivisions, model, device='cpu', filename='subcell_nodes_ellipse.png'):
    """
    Saves a plot of the subcell layout for the ellipse:
      - For n==1, the entire domain is treated as one (partial) cell and the CNN is called.
      - For n > 1, cells fully inside or outside are filled uniformly; partial cells are processed with the CNN.
    Overlays the analytical ellipse boundary and subcell grid lines.
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
                inside_flags = [is_inside_ellipse(x, y) for x, y in zip(corners_x, corners_y)]
                # Also checks the center of the subcell
                center_flag = [is_inside_ellipse(ox, oy)] 
                if all(inside_flags):
                    cell_case = 'inside'
                elif (not any(inside_flags)) and (not center_flag[0]):
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
                exps_x_sub, exps_y_sub, coeffs_sub = make_subcell_ellipse_polynomial(ox, oy, n_subdivisions, device)
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

    # Plot the analytical ellipse boundary using the standard parametric form.
    theta = np.linspace(0, 2*np.pi, 200)
    x_ellipse = C[0] + a * np.cos(theta)*np.cos(angle) - b * np.sin(theta)*np.sin(angle)
    y_ellipse = C[1] + a * np.cos(theta)*np.sin(angle) + b * np.sin(theta)*np.cos(angle)
    plt.plot(x_ellipse, y_ellipse, 'r-', linewidth=2, label='Ellipse Boundary')

    subcell_width = 2.0 / n_subdivisions
    for i in range(n_subdivisions+1):
        coord = -1 + i * subcell_width
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
    """
    Computes the predicted integral areas and relative errors for the ellipse.
    
    Returns:
        error_list (list): List of relative errors (fraction, not percentage).
        refinement_levels (list): List of refinement levels used.
    """
    analytical_area = math.pi * a * b  # Analytical area for the ellipse
    refinement_levels = [1, 2, 4, 8, 16, 32]
    error_list = []  # Relative error for each refinement level.
    area_list = []   # Predicted integral areas.

    print("\nComputing area by subdividing domain and calling CNN per subcell (with full-cell check) for the ellipse:")
    for n in refinement_levels:
        pred_area = compute_h_refined_integral_ellipse(n, model, device=device)
        area_list.append(pred_area)
        rel_error = abs(pred_area - analytical_area) / analytical_area 
        error_list.append(rel_error)
        print(f"  Subcells: {n}x{n}")
        print(f"    Predicted area: {pred_area:.16f}")
        print(f"    Analytical area: {analytical_area:.16f}")
        print(f"    Relative error: {rel_error:.16f}\n")
        
        aggregated_plot_filename = os.path.join(output_folder, f"predicted_nodes_ellipse_n{n}.png")
        save_subcell_nodes_plot_ellipse(n, model, device=device, filename=aggregated_plot_filename)
        print(f"Aggregate subcell plot saved as '{aggregated_plot_filename}'.")

    # Plot Relative Error vs. Element Size (with log scale on both axes).
    element_sizes = [2.0 / n for n in refinement_levels]
    plt.figure(figsize=(8,6))
    plt.plot(element_sizes, error_list, marker='o', linestyle='-', color='b')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Element Size (2 / n) [log scale]")
    plt.ylabel("Relative Error [log scale]")
    plt.title("Relative Error vs. Element Size (Log-Log) for Ellipse")
    plt.grid(True, which='both', ls='--', alpha=0.7)
    plt.savefig(os.path.join(output_folder, "error_vs_element_size_ellipse.png"), dpi=300)
    plt.close()
    print("Relative error vs. element size plot saved as 'error_vs_element_size_ellipse.png'.")

    # Plot Integral Area vs. Refinement Level.
    plt.figure(figsize=(8,6))
    plt.plot(refinement_levels, area_list, marker='o', linestyle='-', color='b', label='Predicted Integral Area')
    plt.axhline(y=analytical_area, color='r', linestyle='--', label='Analytical Area')
    plt.xlabel("Number of Subcells per Dimension (Refinement Level)")
    plt.ylabel("Integral Area")
    plt.title("Integral Area vs. Refinement Level for Ellipse")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "area_vs_refinement_ellipse.png"), dpi=300)
    plt.close()
    print("Integral area plot saved as 'area_vs_refinement_ellipse.png'.")

    return error_list, refinement_levels

###############################################################################
# 7. Main Script: Run everything if executed directly
###############################################################################
def main():
    compute_error_ellipse()

if __name__ == "__main__":
    main()
