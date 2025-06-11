#!/usr/bin/env python

import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import os

from model_hybrid import load_shallow_cnn_model
import utilities  # Must provide utilities.compute_integration(...)

###############################################################################
# 1. Load the FNN Model (following the exact style of the CNN study)
###############################################################################
device = torch.device('cpu')
model_path = r"C:\Git\MasterThesis\Models\Hybrid\Hybrid_V5\fnn_model_weights_v4.pth"
output_folder = r"C:\Git\MasterThesis\Scripts\CNN_hybrid\Hybrid_V5\plt\circle"
os.makedirs(output_folder, exist_ok=True)

node_x_str = """-0.9602898564975362,-0.9602898564975362,-0.9602898564975362,-0.9602898564975362,-0.9602898564975362,-0.9602898564975362,-0.9602898564975362,-0.9602898564975362,
    -0.7966664774136267,-0.7966664774136267,-0.7966664774136267,-0.7966664774136267,-0.7966664774136267,-0.7966664774136267,-0.7966664774136267,-0.7966664774136267,
    -0.5255324099163290,-0.5255324099163290,-0.5255324099163290,-0.5255324099163290,-0.5255324099163290,-0.5255324099163290,-0.5255324099163290,-0.5255324099163290,
    -0.1834346424956498,-0.1834346424956498,-0.1834346424956498,-0.1834346424956498,-0.1834346424956498,-0.1834346424956498,-0.1834346424956498,-0.1834346424956498,
    0.1834346424956499,0.1834346424956499,0.1834346424956499,0.1834346424956499,0.1834346424956499,0.1834346424956499,0.1834346424956499,0.1834346424956499,
    0.5255324099163290,0.5255324099163290,0.5255324099163290,0.5255324099163290,0.5255324099163290,0.5255324099163290,0.5255324099163290,0.5255324099163290,
    0.7966664774136267,0.7966664774136267,0.7966664774136267,0.7966664774136267,0.7966664774136267,0.7966664774136267,0.7966664774136267,0.7966664774136267,
    0.9602898564975362,0.9602898564975362,0.9602898564975362,0.9602898564975362,0.9602898564975362,0.9602898564975362,0.9602898564975362,0.9602898564975362"""

node_y_str = """-0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956499,0.5255324099163290,0.7966664774136267,0.9602898564975362,
    -0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956499,0.5255324099163290,0.7966664774136267,0.9602898564975362,
    -0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956499,0.5255324099163290,0.7966664774136267,0.9602898564975362,
    -0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956499,0.5255324099163290,0.7966664774136267,0.9602898564975362,
    -0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956499,0.5255324099163290,0.7966664774136267,0.9602898564975362,
    -0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956499,0.5255324099163290,0.7966664774136267,0.9602898564975362,
    -0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956499,0.5255324099163290,0.7966664774136267,0.9602898564975362,
    -0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956499,0.5255324099163290,0.7966664774136267,0.9602898564975362"""


model = load_shallow_cnn_model(
    None,
    node_x_str=node_x_str,
    node_y_str=node_y_str
)

# (2) load the state dict onto CPU explicitly
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# (3) then move to your chosen device (in this case CPU)
model.to(device)
model.eval()


###############################################################################
# 2. Helper: Build the Circle Polynomial in a Subcell (radius = 0.4)
#    now normalized exactly as in the CNN study (/h^2)
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
    # Normalize by h^2 exactly as in the CNN version
    c0  = (-1.0 + 6.25*(ox*ox + oy*oy)) / (h*h)
    cX  = (12.5 * ox * h)   / (h*h)    # = 12.5 * ox / h
    cY  = (12.5 * oy * h)   / (h*h)    # = 12.5 * oy / h
    cX2 = (6.25 * h*h)      / (h*h)    # = 6.25
    cY2 = cX2

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
# 4. Subcell-based Integration with Full-Cell Check + FNN fallback
###############################################################################
def compute_h_refined_integral(n_subdivisions, model, device='cpu'):
    """
    Exactly mirrors the CNN study's compute_h_refined_integral,
    but uses the FNN model as fallback on partial cells.
    """
    h = 1.0 / n_subdivisions
    jacobian = h*h
    centers = np.linspace(-1 + h, 1 - h, n_subdivisions)
    total_integral = 0.0

    # reference-cell nodes for integration
    nodes_x_ref = model.nodal_preprocessor.X.unsqueeze(0).to(device)
    nodes_y_ref = model.nodal_preprocessor.Y.unsqueeze(0).to(device)

    for ox in centers:
        for oy in centers:
            # full/empty/partial check
            if n_subdivisions > 1:
                flags = [
                    is_inside_circle(ox + dx*h, oy + dy*h)
                    for dx in (-1,1) for dy in (-1,1)
                ]
                if all(flags):
                    # full cell: integral over reference = 4
                    total_integral += jacobian * 4.0
                    continue
                if not any(flags):
                    # fully outside
                    continue

            # partial cell → use FNN
            exps_x_sub, exps_y_sub, coeffs_sub = make_subcell_circle_polynomial(ox, oy, n_subdivisions, device)
            with torch.no_grad():
                _, _, pred_weights = model(exps_x_sub, exps_y_sub, coeffs_sub)
            pred_weights = pred_weights.view(1, -1)

            # integrate on reference cell and map via jacobian
            subint = utilities.compute_integration(
                nodes_x_ref, nodes_y_ref, pred_weights, lambda x, y: 1.0
            )[0].item()
            total_integral += jacobian * subint

    return total_integral


###############################################################################
# 5. Single Plot of Partial‐Cell Nodes (modified for FNN)
###############################################################################
def save_subcell_nodes_plot(n_subdivisions, model, device='cpu', filename='subcell_nodes.png'):
    """
    Same layout + styling as the CNN version, but for FNN:
      - full inside: green
      - full outside: gray
      - partial: scatter FNN‐predicted nodes & weights
    Overlays the true circle boundary and grid.
    """
    h = 1.0 / n_subdivisions
    centers = np.linspace(-1 + h, 1 - h, n_subdivisions)

    partial_x_all = []
    partial_y_all = []
    partial_w_all = []

    plt.figure(figsize=(8,8))

    for ox in centers:
        for oy in centers:
            # classify cell
            if n_subdivisions == 1:
                case = 'partial'
            else:
                flags = [
                    is_inside_circle(ox + dx*h, oy + dy*h)
                    for dx in (-1,1) for dy in (-1,1)
                ]
                if all(flags):
                    case = 'inside'
                elif not any(flags):
                    case = 'outside'
                else:
                    case = 'partial'

            if case == 'inside':
                rect = plt.Rectangle(
                    (ox - h, oy - h), 2*h, 2*h,
                    facecolor='lightgreen', edgecolor='blue',
                    linestyle='--', alpha=0.5
                )
                plt.gca().add_patch(rect)

            elif case == 'outside':
                rect = plt.Rectangle(
                    (ox - h, oy - h), 2*h, 2*h,
                    facecolor='lightgray', edgecolor='blue',
                    linestyle='--', alpha=0.3
                )
                plt.gca().add_patch(rect)

            else:  # partial
                exps_x_sub, exps_y_sub, coeffs_sub = make_subcell_circle_polynomial(ox, oy, n_subdivisions, device)
                with torch.no_grad():
                    px, py, pw = model(exps_x_sub, exps_y_sub, coeffs_sub)
                x_phys = (px * h + ox).cpu().numpy().ravel()
                y_phys = (py * h + oy).cpu().numpy().ravel()
                w_phys = pw.cpu().numpy().ravel()

                partial_x_all.append(x_phys)
                partial_y_all.append(y_phys)
                partial_w_all.append(w_phys)

    if partial_x_all:
        X = np.concatenate(partial_x_all)
        Y = np.concatenate(partial_y_all)
        W = np.concatenate(partial_w_all)
        sc = plt.scatter(X, Y, c=W, cmap='viridis', s=10, edgecolors='k')
        plt.colorbar(sc, label="Predicted Weight")

    # draw true boundary
    theta = np.linspace(0, 2*np.pi, 200)
    plt.plot(0.4*np.cos(theta), 0.4*np.sin(theta), 'r-', linewidth=2, label='Circle Boundary')

    # grid lines
    for i in range(n_subdivisions+1):
        coord = -1 + i*(2*h)
        plt.axvline(x=coord, linestyle='--', color='blue', linewidth=0.5)
        plt.axhline(y=coord, linestyle='--', color='blue', linewidth=0.5)

    plt.title(f"Subcell‐based Predicted Nodes (n={n_subdivisions})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect('equal', 'box')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
def plot_all_subcell_nodes_landscape(refinement_levels, model, device='cpu', filename='all_subcell_nodes_landscape.png'):
    """
    Landscape plot of all refinement levels: arranges subcell prediction plots side by side for each n
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    n_levels = len(refinement_levels)
    # create one row of subplots, one column per level
    fig, axes = plt.subplots(1, n_levels, figsize=(5 * n_levels, 5), squeeze=False)

    # prepare true circle boundary
    theta = np.linspace(0, 2 * np.pi, 200)
    x_circle = 0.4 * np.cos(theta)
    y_circle = 0.4 * np.sin(theta)

    for ax, n in zip(axes[0], refinement_levels):
        h = 1.0 / n
        centers = np.linspace(-1 + h, 1 - h, n)
        partial_x, partial_y, partial_w = [], [], []

        for ox in centers:
            for oy in centers:
                # determine cell case
                if n == 1:
                    case = 'partial'
                else:
                    corners = [(ox - h, oy - h), (ox - h, oy + h), (ox + h, oy - h), (ox + h, oy + h)]
                    flags = [is_inside_circle(xc, yc) for xc, yc in corners]
                    case = 'inside' if all(flags) else 'outside' if not any(flags) else 'partial'

                if case == 'inside':
                    rect = plt.Rectangle((ox - h, oy - h), 2 * h, 2 * h,
                                         facecolor='lightgreen', edgecolor='blue', alpha=0.5, linestyle='--')
                    ax.add_patch(rect)
                elif case == 'outside':
                    rect = plt.Rectangle((ox - h, oy - h), 2 * h, 2 * h,
                                         facecolor='lightgray', edgecolor='blue', alpha=0.3, linestyle='--')
                    ax.add_patch(rect)
                else:
                    exps_x_sub, exps_y_sub, coeffs_sub = make_subcell_circle_polynomial(ox, oy, n, device)
                    with torch.no_grad():
                        px, py, pw = model(exps_x_sub, exps_y_sub, coeffs_sub)
                    x_phys = (ox + h * px).cpu().numpy().ravel()
                    y_phys = (oy + h * py).cpu().numpy().ravel()
                    w_phys = pw.cpu().numpy().ravel()
                    partial_x.append(x_phys)
                    partial_y.append(y_phys)
                    partial_w.append(w_phys)

        # scatter partial predictions
        if partial_x:
            X = np.concatenate(partial_x)
            Y = np.concatenate(partial_y)
            W = np.concatenate(partial_w)
            sc = ax.scatter(X, Y, c=W, cmap='viridis', s=10, edgecolors='k')

        # draw true boundary
        ax.plot(x_circle, y_circle, 'r-', linewidth=2)
        ax.set_title(f"n = {n}")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', linewidth=0.5)

    # add single colorbar above all
    if 'sc' in locals():
        cbar = fig.colorbar(sc, ax=axes[0].tolist(), orientation='horizontal', fraction=0.04, pad=0.12, location='top', aspect=40)
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.set_label('Predicted Weight')

    plt.subplots_adjust(top=0.85, bottom=0.10)
    plt.tight_layout(rect=[0, 0.10, 1, 0.85])
    # ensure output folder exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.close()


###############################################################################
# 6. Compute error_list and area_list (same printing + plotting style)
###############################################################################
def compute_error_circle():
    """
    Computes predicted integral and relative error for each refinement.
    """
    analytical_area   = math.pi * (0.4**2)
    refinement_levels = [1, 2, 4, 8, 16]
    area_list         = []
    error_list        = []

    print("\nComputing area by subdividing domain and calling FNN per subcell (with full-cell check):")
    for n in refinement_levels:
        pred_area = compute_h_refined_integral(n, model, device=device)
        rel_error = abs(pred_area - analytical_area) / analytical_area

        area_list.append(pred_area)
        error_list.append(rel_error)

        print(f"  Subcells: {n}x{n}")
        print(f"    Predicted area : {pred_area:.16f}")
        print(f"    Analytical area: {analytical_area:.16f}")
        print(f"    Relative error : {rel_error:.16f}\n")

        png = os.path.join(output_folder, f"predicted_nodes_n{n}.png")
        save_subcell_nodes_plot(n, model, device=device, filename=png)
        print(f"Aggregate subcell plot saved as '{png}'.")

    # --- Relative error vs element size (log-log) ---
    element_sizes = [2.0/n for n in refinement_levels]
    plt.figure(figsize=(8,6))
    plt.plot(element_sizes, error_list, marker='o', linestyle='-', color='b')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Element Size (2 / n) [log scale]")
    plt.ylabel("Relative Error [log scale]")
    plt.title("Relative Error vs. Element Size (Log-Log) v2")
    plt.grid(True, which='both', ls='--', alpha=0.7)
    err_fn = os.path.join(output_folder, "error_vs_element_size_v2.png")
    plt.savefig(err_fn, dpi=300)
    plt.close()
    print(f"Relative error vs. element size plot saved as '{err_fn}'.")

    # --- Integral area vs refinement ---
    plt.figure(figsize=(8,6))
    plt.plot(refinement_levels, area_list, marker='o', linestyle='-', color='b',
             label='Predicted Integral Area')
    plt.axhline(y=analytical_area, color='r', linestyle='--', label='Analytical Area')
    plt.xlabel("Number of Subcells per Dimension (Refinement Level)")
    plt.ylabel("Integral Area")
    plt.title("Integral Area vs. Refinement Level")
    plt.legend()
    plt.grid(True)
    area_fn = os.path.join(output_folder, "area_vs_refinement.png")
    plt.savefig(area_fn, dpi=300)
    plt.close()
    print(f"Integral area plot saved as '{area_fn}'.")

    return error_list, refinement_levels



###############################################################################
# 7. Main Script: Run everything if executed directly
###############################################################################
def main():
    error_list, refinement_levels = compute_error_circle()
    # Generate combined landscape plot
    landscape_fn = os.path.join(output_folder, "all_subcell_nodes_landscape.png")
    plot_all_subcell_nodes_landscape(refinement_levels, model, device=device, filename=landscape_fn)
    print(f"Combined landscape plot saved as '{landscape_fn}'")



if __name__ == "__main__":
    main()
