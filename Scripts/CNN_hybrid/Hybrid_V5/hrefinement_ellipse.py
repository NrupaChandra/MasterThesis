#!/usr/bin/env python

import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import os

from model_hybrid import load_shallow_cnn_model
import utilities  # Must provide utilities.compute_integration(...)

###############################################################################
# Ellipse parameters
###############################################################################
a = 0.2          # first semi-axis
b = 0.35         # second semi-axis
C = (0.4, 0.6)   # center of the ellipse (C_x, C_y)
angle = np.pi/3  # rotation angle in radians (60°)

###############################################################################
# 1. Load the FNN Model (exactly as in the circle script)
###############################################################################
device = torch.device('cpu')
model_path = r"C:\Git\MasterThesis\Models\Hybrid\Hybrid_V5\fnn_model_weights_v4.pth"
output_folder = r"C:\Git\MasterThesis\Scripts\CNN_hybrid\Hybrid_V5\plt\ellipse"
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

# (2) load the weights onto CPU
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# (3) then move to eval device
model.to(device)
model.eval()


###############################################################################
# 2. Helper: Build the Ellipse Polynomial (normalized exactly like circle)
###############################################################################
def make_subcell_ellipse_polynomial(ox, oy, n, device='cpu'):
    """
    f_phys(x,y) = (X'^2)/a^2 + (Y'^2)/b^2 - 1, on subcell center (ox,oy), 
    then normalize by 1/h^2 so the network always sees a reference-cell poly.
    """
    h = 1.0 / n
    # rotation + shift coefficients
    A_x =  h * math.cos(angle)
    A_y =  h * math.sin(angle)
    A_0 = (ox - C[0]) * math.cos(angle) + (oy - C[1]) * math.sin(angle)
    B_x = -h * math.sin(angle)
    B_y =  h * math.cos(angle)
    B_0 = -(ox - C[0]) * math.sin(angle) + (oy - C[1]) * math.cos(angle)

    # expand and normalize by h^2
    c0  = ((A_0**2)/a**2 + (B_0**2)/b**2 - 1.0) / (h*h)
    cX  = ((2*A_x*A_0)/a**2 + (2*B_x*B_0)/b**2)  / (h*h)
    cY  = ((2*A_y*A_0)/a**2 + (2*B_y*B_0)/b**2)  / (h*h)
    cX2 = ((A_x**2)/a**2 + (B_x**2)/b**2)        / (h*h)
    cXY = ((2*A_x*A_y)/a**2 + (2*B_x*B_y)/b**2)  / (h*h)
    cY2 = ((A_y**2)/a**2 + (B_y**2)/b**2)        / (h*h)

    exps_x = torch.tensor([[0,1,0,2,1,0]], dtype=torch.float32, device=device)
    exps_y = torch.tensor([[0,0,1,0,1,2]], dtype=torch.float32, device=device)
    coeffs = torch.tensor([[c0, cX, cY, cX2, cXY, cY2]],
                          dtype=torch.float32, device=device)
    return exps_x, exps_y, coeffs


###############################################################################
# 3. Ellipse "Inside" Checker (physical-space test)
###############################################################################
def is_inside_ellipse(x, y):
    x_shift = x - C[0]
    y_shift = y - C[1]
    Xp = x_shift * math.cos(angle) + y_shift * math.sin(angle)
    Yp = -x_shift * math.sin(angle) + y_shift * math.cos(angle)
    return (Xp*Xp)/(a*a) + (Yp*Yp)/(b*b) <= 1.0


###############################################################################
# 4. FNN-fallback integration over ellipse subcells (same logic as circle)
###############################################################################
def compute_h_refined_integral_ellipse(n_subdivisions, model, device='cpu'):
    h = 1.0 / n_subdivisions
    jacobian = h*h
    centers = np.linspace(-1 + h, 1 - h, n_subdivisions)
    total = 0.0

    # reference-cell nodes
    nodes_x_ref = model.nodal_preprocessor.X.unsqueeze(0).to(device)
    nodes_y_ref = model.nodal_preprocessor.Y.unsqueeze(0).to(device)

    for ox in centers:
        for oy in centers:
            # full/empty cell test
            if n_subdivisions > 1:
                corners = [(ox + dx*h, oy + dy*h) for dx in (-1,1) for dy in (-1,1)]
                flags = [is_inside_ellipse(xc, yc) for xc, yc in corners]
                if all(flags):
                    total += jacobian * 4.0
                    continue
                if not any(flags):
                    continue

            # partial → FNN fallback
            exps_x_sub, exps_y_sub, coeffs_sub = make_subcell_ellipse_polynomial(ox, oy, n_subdivisions, device)
            with torch.no_grad():
                _, _, pred_w = model(exps_x_sub, exps_y_sub, coeffs_sub)
            pred_w = pred_w.view(1, -1)

            subint = utilities.compute_integration(
                nodes_x_ref, nodes_y_ref, pred_w, lambda x, y: 1.0
            )[0].item()
            total += jacobian * subint

    return total


###############################################################################
# 5. Plot the partial-cell nodes for ellipse (mirroring circle code)
###############################################################################
def save_subcell_nodes_plot_ellipse(n_subdivisions, model, device='cpu', filename='subcell_nodes_ellipse.png'):
    h = 1.0 / n_subdivisions
    centers = np.linspace(-1 + h, 1 - h, n_subdivisions)
    partial_x, partial_y, partial_w = [], [], []

    plt.figure(figsize=(8,8))
    for ox in centers:
        for oy in centers:
            # classify
            if n_subdivisions == 1:
                case = 'partial'
            else:
                corners = [(ox + dx*h, oy + dy*h) for dx in (-1,1) for dy in (-1,1)]
                flags = [is_inside_ellipse(xc, yc) for xc, yc in corners]
                if all(flags):
                    case = 'inside'
                elif not any(flags):
                    case = 'outside'
                else:
                    case = 'partial'

            if case == 'inside':
                rect = plt.Rectangle((ox-h,oy-h), 2*h,2*h,
                                     facecolor='lightgreen', edgecolor='blue',
                                     linestyle='--', alpha=0.5)
                plt.gca().add_patch(rect)
            elif case == 'outside':
                rect = plt.Rectangle((ox-h,oy-h), 2*h,2*h,
                                     facecolor='lightgray', edgecolor='blue',
                                     linestyle='--', alpha=0.3)
                plt.gca().add_patch(rect)
            else:
                exps_x_sub, exps_y_sub, coeffs_sub = make_subcell_ellipse_polynomial(ox, oy, n_subdivisions, device)
                with torch.no_grad():
                    px, py, pw = model(exps_x_sub, exps_y_sub, coeffs_sub)
                X_phys = (px * h + ox).cpu().numpy().ravel()
                Y_phys = (py * h + oy).cpu().numpy().ravel()
                W_phys = pw .cpu().numpy().ravel()
                partial_x.append(X_phys)
                partial_y.append(Y_phys)
                partial_w.append(W_phys)

    if partial_x:
        X = np.concatenate(partial_x)
        Y = np.concatenate(partial_y)
        W = np.concatenate(partial_w)
        sc = plt.scatter(X, Y, c=W, cmap='viridis', s=10, edgecolors='k')
        plt.colorbar(sc, label="Predicted Weight")

    # draw analytic ellipse
    θ = np.linspace(0,2*np.pi,200)
    Xe = C[0] + a*np.cos(θ)*np.cos(angle) - b*np.sin(θ)*np.sin(angle)
    Ye = C[1] + a*np.cos(θ)*np.sin(angle) + b*np.sin(θ)*np.cos(angle)
    plt.plot(Xe, Ye, 'r-', linewidth=2, label='Ellipse Boundary')

    # grid
    for i in range(n_subdivisions+1):
        c = -1 + 2*i*h
        plt.axvline(c, linestyle='--', color='blue', linewidth=0.5)
        plt.axhline(c, linestyle='--', color='blue', linewidth=0.5)

    plt.title(f"Subcell-based Predicted Nodes for Ellipse (n={n_subdivisions})")
    plt.xlabel("X"); plt.ylabel("Y")
    plt.gca().set_aspect('equal','box')
    plt.xlim(-1,1); plt.ylim(-1,1)
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(filename, dpi=300); plt.close()


###############################################################################
# 6. Compute error_list and area_list (identical style to circle code)
###############################################################################
def compute_error_ellipse():
    analytical = math.pi * a * b
    levels     = [1,2,4,8,16]
    errors, areas = [], []

    print("\nComputing area by subdividing domain and calling FNN per subcell (with full-cell check) for the ellipse:")
    for n in levels:
        area = compute_h_refined_integral_ellipse(n, model, device=device)
        rel  = abs(area - analytical) / analytical
        areas.append(area); errors.append(rel)

        print(f"  Subcells: {n}×{n}")
        print(f"    Predicted area : {area: .16f}")
        print(f"    Analytical area: {analytical: .16f}")
        print(f"    Relative error : {rel: .16f}\n")

        png = os.path.join(output_folder, f"predicted_nodes_ellipse_n{n}.png")
        save_subcell_nodes_plot_ellipse(n, model, device=device, filename=png)
        print(f"Aggregate subcell plot saved as '{png}'.")

    # error vs element size
    esizes = [2.0/n for n in levels]
    plt.figure(figsize=(8,6))
    plt.plot(esizes, errors, 'o-', color='b')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel("Element Size (2/n) [log]"); plt.ylabel("Relative Error [log]")
    plt.title("Relative Error vs. Element Size (Log-Log) for Ellipse")
    plt.grid(True, which='both', ls='--', alpha=0.7)
    fn = os.path.join(output_folder, "error_vs_element_size_ellipse.png")
    plt.savefig(fn, dpi=300); plt.close()
    print(f"Relative error vs. element size plot saved as '{fn}'.")

    # area vs refinement
    plt.figure(figsize=(8,6))
    plt.plot(levels, areas, 'o-', color='b', label='Predicted Integral Area')
    plt.axhline(y=analytical, color='r', linestyle='--', label='Analytical Area')
    plt.xlabel("Refinement Level (n subdivisions per side)")
    plt.ylabel("Integral Area"); plt.title("Integral Area vs. Refinement Level for Ellipse")
    plt.legend(); plt.grid(True)
    fn2 = os.path.join(output_folder, "area_vs_refinement_ellipse.png")
    plt.savefig(fn2, dpi=300); plt.close()
    print(f"Integral area plot saved as '{fn2}'.")

    return errors, levels


###############################################################################
# 7. Main
###############################################################################
def main():
    compute_error_ellipse()

if __name__ == "__main__":
    main()
