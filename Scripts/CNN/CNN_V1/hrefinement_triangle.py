#!/usr/bin/env python

import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import os

from model_cnn import load_shallow_cnn_model
import utilities  # Must provide utilities.compute_integration(...)

###############################################################################
# New Shape parameters and level-set function
###############################################################################

def triangle(x, y):
    return -x - y + 0.5 

###############################################################################
# 1. Load the CNN Model
###############################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = r"C:\Git\MasterThesis\Models\CNN\CNN_V1\cnn_model_weights_v1.0.pth"
output_folder = r"C:\Git\MasterThesis\Scripts\CNN\CNN_V1\plt\Triangle"

model = load_shallow_cnn_model(
    weights_path=model_path,
    num_nodes=1225,       
    domain=(-1, 1),      
    dropout_rate=0.0
)
model.to(device)
model.eval()

###############################################################################
# 2. Helper: Build the New Shape Polynomial in a Subcell
###############################################################################
def make_subcell_triangle_polynomial(ox, oy, n, device='cpu'):
    """
    Constructs the polynomial representation for the new shape level-set function
    F(x, y) = -x - y + 0.5 in a subcell.

    In the subcell, the mapping from the local coordinates (X, Y) to the physical
    coordinates (x, y) is given by:
        x = (1/n) * X + ox
        y = (1/n) * Y + oy,
    where n is the number of subdivisions and sub_half = 1/n.

    After substitution into F(x, y):
        F(x,y) = -((1/n)*X + ox) - ((1/n)*Y + oy) + 0.5
               = (0.5 - ox - oy) - (1/n)*X - (1/n)*Y.
    
    The polynomial is then represented with the monomials:
        1,  X,  Y,  X^2,  X*Y,  Y^2

    Coefficients:
        coeff_const = 0.5 - ox - oy
        coeff_X     = -(1/n)
        coeff_Y     = -(1/n)
        coeff_X2    = 0.0
        coeff_XY    = 0.0
        coeff_Y2    = 0.0

    Returns:
        exps_x (torch.Tensor): Exponents for X in each monomial.
        exps_y (torch.Tensor): Exponents for Y in each monomial.
        coeffs (torch.Tensor): Tensor of coefficients for each monomial.
    """
    sub_half = 1.0 / n  # The scaling factor for the subcell.
    
    # Define coefficients based on the expansion above:
    coeff_const = 0.5 - ox - oy
    coeff_X     = -sub_half
    coeff_Y     = -sub_half
    coeff_X2    = 0.0
    coeff_XY    = 0.0
    coeff_Y2    = 0.0

    # Monomials are defined as: 1, X, Y, X^2, X*Y, Y^2.
    exps_x = torch.tensor([[0, 1, 0, 2, 1, 0]], dtype=torch.float32, device=device)
    exps_y = torch.tensor([[0, 0, 1, 0, 1, 2]], dtype=torch.float32, device=device)
    
    coeffs = torch.tensor([[coeff_const, coeff_X, coeff_Y, coeff_X2, coeff_XY, coeff_Y2]], 
                          dtype=torch.float32, device=device)

    return exps_x, exps_y, coeffs

###############################################################################
# 3. New Shape "Inside" Checker
###############################################################################
def is_inside_triangle(x, y):
    """
    Returns True if the point (x, y) lies inside the new shape,
    i.e., if F(x,y) = x + y - 0.5 - 0.3*x*y <= 0.
    """
    return triangle(x, y) <= 0.0

###############################################################################
# 4. Subcell-based Integration with Full-Cell Check (for the New Shape)
###############################################################################
def compute_h_refined_integral_triangle(n_subdivisions, model, device='cpu'):
    """
    For each subcell:
      - For n > 1, if the subcell is entirely inside the new shape, assign full area;
        if entirely outside, assign zero; otherwise, call the CNN.
      - For n == 1, always call the CNN so that the entire domain is used.
    
    The subcell's integration value (obtained directly or via the CNN) is then 
    multiplied by the Jacobian and summed over all subcells.
    """
    subcell_half = 1.0 / n_subdivisions  # half-width (in physical units)
    jacobian = subcell_half ** 2          # Jacobian factor
    centers = np.linspace(-1 + subcell_half, 1 - subcell_half, n_subdivisions)
    total_integral = 0.0

    nodes_x_ref = model.nodal_preprocessor.X.unsqueeze(0).to(device)
    nodes_y_ref = model.nodal_preprocessor.Y.unsqueeze(0).to(device)

    for ox in centers:
        for oy in centers:
            if n_subdivisions == 1:
                exps_x_sub, exps_y_sub, coeffs_sub = make_subcell_triangle_polynomial(ox, oy, n_subdivisions, device)
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
                inside_flags = [is_inside_triangle(x, y) for x, y in zip(corners_x, corners_y)]
                center_flag = [is_inside_triangle(ox, oy)]
                if all(inside_flags):
                    subcell_integral = 4.0
                elif not any(inside_flags) and not center_flag[0]:
                    subcell_integral = 0.0
                else:
                    exps_x_sub, exps_y_sub, coeffs_sub = make_subcell_triangle_polynomial(ox, oy, n_subdivisions, device)
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
# 5. Single Plot of All Subcells, Skipping CNN for Fully Inside (New Shape Version)
###############################################################################
def save_subcell_nodes_plot_triangle(n_subdivisions, model, device='cpu', filename='subcell_nodes_triangle.png'):
    """
    Saves a plot showing the subcell layout for the new shape:
      - For n==1, the entire domain is treated as one (partial) cell.
      - For n > 1, cells completely inside or outside are filled uniformly 
        while partial cells (neither fully inside nor outside) are processed with the CNN.
    
    The new shape boundary is overlaid using a contour plot of F(x,y)=0.
    """
    subcell_half = 1.0 / n_subdivisions
    centers = np.linspace(-1 + subcell_half, 1 - subcell_half, n_subdivisions)

    partial_x_all = []
    partial_y_all = []
    partial_w_all = []

    nodes_x_ref = model.nodal_preprocessor.X.unsqueeze(0).to(device)
    nodes_y_ref = model.nodal_preprocessor.Y.unsqueeze(0).to(device)

    plt.figure(figsize=(8, 8))

    for ox in centers:
        for oy in centers:
            if n_subdivisions == 1:
                cell_case = 'partial'
            else:
                corners_x = [ox - subcell_half, ox - subcell_half, ox + subcell_half, ox + subcell_half]
                corners_y = [oy - subcell_half, oy + subcell_half, oy - subcell_half, oy + subcell_half]
                inside_flags = [is_inside_triangle(x, y) for x, y in zip(corners_x, corners_y)]
                center_flag = [is_inside_triangle(ox, oy)]
                if all(inside_flags):
                    cell_case = 'inside'
                elif (not any(inside_flags)) and (not center_flag[0]):
                    cell_case = 'outside'
                else:
                    cell_case = 'partial'

            if cell_case == 'inside':
                rect = plt.Rectangle(
                    (ox - subcell_half, oy - subcell_half),
                    2 * subcell_half, 2 * subcell_half,
                    facecolor='lightgreen', edgecolor='blue', alpha=0.5, linestyle='--'
                )
                plt.gca().add_patch(rect)
            elif cell_case == 'outside':
                rect = plt.Rectangle(
                    (ox - subcell_half, oy - subcell_half),
                    2 * subcell_half, 2 * subcell_half,
                    facecolor='lightgray', edgecolor='blue', alpha=0.3, linestyle='--'
                )
                plt.gca().add_patch(rect)
            else:
                exps_x_sub, exps_y_sub, coeffs_sub = make_subcell_triangle_polynomial(ox, oy, n_subdivisions, device)
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
            partial_x_all, partial_y_all, c=partial_w_all,
            cmap='viridis', s=10, edgecolors='k'
        )
        plt.colorbar(sc, label="Predicted Weight")

    # Plot the new shape boundary using a contour (zero level set)
    X_grid, Y_grid = np.meshgrid(np.linspace(-1, 1, 400), np.linspace(-1, 1, 400))
    F_grid = triangle(X_grid, Y_grid)
    plt.contour(X_grid, Y_grid, F_grid, levels=[0], colors='r', linewidths=2)

    subcell_width = 2.0 / n_subdivisions
    for i in range(n_subdivisions + 1):
        coord = -1 + i * subcell_width
        plt.axvline(x=coord, color='blue', linestyle='--', linewidth=0.5)
        plt.axhline(y=coord, color='blue', linestyle='--', linewidth=0.5)

    plt.title(f"Subcell-based Predicted Nodes for New Shape (n={n_subdivisions} per dimension)")
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
# 6. Compute Error and Area for the New Shape
###############################################################################
def compute_error_triangle():
   
    analytical_area = 0.0  
    analytical_area = 0.5*1.5*1.5

    refinement_levels = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    error_list = []
    area_list = []

    print("\nComputing area by subdividing domain and calling CNN per subcell (with full-cell check) for the new shape:")
    for n in refinement_levels:
        pred_area = compute_h_refined_integral_triangle(n, model, device=device)
        area_list.append(pred_area)
        rel_error = abs(pred_area - analytical_area) / analytical_area
        error_list.append(rel_error)
        print(f"  Subcells: {n}x{n}")
        print(f"    Predicted area: {pred_area:.16f}")
        print(f"    Analytical area: {analytical_area:.16f}")
        print(f"    Relative error: {rel_error:.16f}\n")
        
        aggregated_plot_filename = os.path.join(output_folder, f"predicted_nodes_triangle_n{n}.png")
        save_subcell_nodes_plot_triangle(n, model, device=device, filename=aggregated_plot_filename)
        print(f"Aggregate subcell plot saved as '{aggregated_plot_filename}'.")

    # Plot Relative Error vs. Element Size (log-log scale)
    element_sizes = [2.0 / n for n in refinement_levels]
    plt.figure(figsize=(8, 6))
    plt.plot(element_sizes, error_list, marker='o', linestyle='-', color='b')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Element Size (2/n) [log scale]")
    plt.ylabel("Relative Error [log scale]")
    plt.title("Relative Error vs. Element Size (Log-Log) for New Shape")
    plt.grid(True, which='both', ls='--', alpha=0.7)
    plt.savefig(os.path.join(output_folder, "error_vs_element_size_triangle.png"), dpi=300)
    plt.close()
    print("Relative error vs. element size plot saved as 'error_vs_element_size_triangle.png'.")

    # Plot Integral Area vs. Refinement Level.
    plt.figure(figsize=(8, 6))
    plt.plot(refinement_levels, area_list, marker='o', linestyle='-', color='b', label='Predicted Integral Area')
    plt.axhline(y=analytical_area, color='r', linestyle='--', label='Analytical Area')
    plt.xlabel("Number of Subcells per Dimension (Refinement Level)")
    plt.ylabel("Integral Area")
    plt.title("Integral Area vs. Refinement Level for New Shape")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "area_vs_refinement_triangle.png"), dpi=300)
    plt.close()
    print("Integral area plot saved as 'area_vs_refinement_triangle.png'.")

    return error_list, refinement_levels

###############################################################################
# 7. Main Script: Run everything if executed directly
###############################################################################
def main():
    compute_error_triangle()

if __name__ == "__main__":
    main()
