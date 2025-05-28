#!/usr/bin/env python

import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import os

from model_gcnn import load_gnn_model, NodalPreprocessor
from torch_geometric.nn import knn_graph
import utilities  # Must provide utilities.compute_integration(...)

###############################################################################
# Ellipse parameters
###############################################################################
a, b     = 0.2, 0.35
C_x, C_y = 0.4, 0.6
angle    = math.pi/3

###############################################################################
# 1. Load GCNN Model & Build Graph
###############################################################################
device        = torch.device('cpu')
model_path    = r"C:\Git\MasterThesis\Models\GCN\GCN_v3\gcnn_model_weights_v3.pth"
output_folder = r"C:\Git\MasterThesis\Scripts\GCNN\plt\ellipse"
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

preprocessor = NodalPreprocessor(node_x_str, node_y_str).to(device)
pos = torch.stack([preprocessor.X, preprocessor.Y], dim=1).to(device)
edge_index = knn_graph(pos, k=4, loop=False).to(device)

model = load_gnn_model(3, 64, 5, 0.0, device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

###############################################################################
# 2. Build subcell‐ellipse polynomial (normalized by h^2)
###############################################################################
def make_subcell_ellipse_polynomial(ox, oy, n):
    h = 1.0 / n
    ca, sa = math.cos(angle), math.sin(angle)

    A_x, A_y =  h*ca,    h*sa
    A_0      = (ox - C_x)*ca + (oy - C_y)*sa
    B_x, B_y = -h*sa,    h*ca
    B_0      = -(ox - C_x)*sa + (oy - C_y)*ca

    # divide by h^2
    c0  = ((A_0**2)/a**2 + (B_0**2)/b**2 - 1.0)/(h*h)
    cX  = ((2*A_x*A_0)/a**2 + (2*B_x*B_0)/b**2)/(h*h)
    cY  = ((2*A_y*A_0)/a**2 + (2*B_y*B_0)/b**2)/(h*h)
    cX2 = ((A_x**2)/a**2 + (B_x**2)/b**2)/(h*h)
    cXY = ((2*A_x*A_y)/a**2 + (2*B_x*B_y)/b**2)/(h*h)
    cY2 = ((A_y**2)/a**2 + (B_y**2)/b**2)/(h*h)

    exps_x = torch.tensor([[0,1,0,2,1,0]], dtype=torch.float32, device=device)
    exps_y = torch.tensor([[0,0,1,0,1,2]], dtype=torch.float32, device=device)
    coeffs = torch.tensor([[c0, cX, cY, cX2, cXY, cY2]], dtype=torch.float32, device=device)
    return exps_x, exps_y, coeffs

###############################################################################
# 3. Ellipse “inside” checker (now defines its own cos/sin!)
###############################################################################
def is_inside_ellipse(x, y):
    xs, ys = x - C_x, y - C_y
    ca, sa = math.cos(angle), math.sin(angle)
    Xp = xs*ca + ys*sa
    Yp = -xs*sa + ys*ca
    return (Xp*Xp)/(a*a) + (Yp*Yp)/(b*b) <= 1.0

###############################################################################
# 4. h‐refinement integration w/ full/empty test + GCNN fallback
###############################################################################
def compute_h_refined_integral_ellipse(n):
    h   = 1.0 / n
    jac = h*h
    h2  = 0.5*h
    centers = np.linspace(-1+h, 1-h, n)
    total    = 0.0

    for ox in centers:
        for oy in centers:
            if n > 1:
                corners    = [(ox+dx*h, oy+dy*h) for dx in(-1,1) for dy in(-1,1)]
                corner_flg = [is_inside_ellipse(xc,yc) for xc,yc in corners]
                center_flg = is_inside_ellipse(ox, oy)

                if all(corner_flg):
                    total += jac * 4.0
                    continue
                if not any(corner_flg) and not center_flg:
                    continue

            ex, ey, cf = make_subcell_ellipse_polynomial(ox, oy, n)
            with torch.no_grad():
                nodal  = preprocessor(ex, ey, cf)[0]
                feats  = torch.cat([pos, nodal.unsqueeze(1)], dim=1)
                shifts, weights = model(feats, edge_index)

            # map reference shifts into physical subcell
            x_phys = ox + h*pos[:,0] + h2*shifts[:,0]
            y_phys = oy + h*pos[:,1] + h2*shifts[:,1]

            subint = utilities.compute_integration(
                x_phys.unsqueeze(0),
                y_phys.unsqueeze(0),
                weights.unsqueeze(0),
                lambda x,y: 1.0
            )[0].item()
            total += jac * subint

    return total

###############################################################################
# 5. (Optional) Plot partial‐cell nodes
###############################################################################
def save_subcell_nodes_plot_ellipse(n):
    h, h2 = 1.0/n, 0.5/n
    centers = np.linspace(-1+h, 1-h, n)
    Xs, Ys, Ws = [], [], []

    plt.figure(figsize=(8,8))
    for ox in centers:
        for oy in centers:
            if n == 1:
                case = 'partial'
            else:
                corners    = [(ox+dx*h, oy+dy*h) for dx in(-1,1) for dy in(-1,1)]
                corner_flg = [is_inside_ellipse(xc,yc) for xc,yc in corners]
                center_flg = is_inside_ellipse(ox, oy)
                if all(corner_flg):
                    case = 'inside'
                elif not any(corner_flg) and not center_flg:
                    case = 'outside'
                else:
                    case = 'partial'

            if case=='inside':
                plt.gca().add_patch(plt.Rectangle((ox-h,oy-h),2*h,2*h,
                    facecolor='lightgreen', edgecolor='blue', linestyle='--', alpha=0.5))
            elif case=='outside':
                plt.gca().add_patch(plt.Rectangle((ox-h,oy-h),2*h,2*h,
                    facecolor='lightgray', edgecolor='blue', linestyle='--', alpha=0.3))
            else:
                ex, ey, cf = make_subcell_ellipse_polynomial(ox, oy, n)
                with torch.no_grad():
                    nodal  = preprocessor(ex, ey, cf)[0]
                    feats  = torch.cat([pos, nodal.unsqueeze(1)], dim=1)
                    shifts, weights = model(feats, edge_index)

                pts = torch.stack([
                    ox + h*pos[:,0] + h2*shifts[:,0],
                    oy + h*pos[:,1] + h2*shifts[:,1]
                ], dim=1).cpu().numpy()

                Xs.append(pts[:,0]); Ys.append(pts[:,1]); Ws.append(weights.cpu().numpy())

    if Xs:
        X = np.concatenate(Xs); Y = np.concatenate(Ys); W = np.concatenate(Ws)
        sc=plt.scatter(X, Y, c=W, cmap='viridis', s=10, edgecolors='k')
        plt.colorbar(sc, label="Predicted Weight")

    θ = np.linspace(0,2*np.pi,200)
    Xe= C_x + a*np.cos(θ)*math.cos(angle) - b*np.sin(θ)*math.sin(angle)
    Ye= C_y + a*np.cos(θ)*math.sin(angle) + b*np.sin(θ)*math.cos(angle)
    plt.plot(Xe,Ye,'r-',lw=2,label='Ellipse Boundary')

    for i in range(n+1):
        c = -1 + 2*i*h
        plt.axvline(c,ls='--',color='blue',lw=0.5)
        plt.axhline(c,ls='--',color='blue',lw=0.5)

    plt.title(f"Subcell‐based Predicted Nodes for Ellipse (n={n})")
    plt.xlabel("X"); plt.ylabel("Y")
    plt.gca().set_aspect('equal','box')
    plt.xlim(-1,1); plt.ylim(-1,1)
    plt.legend(); plt.grid(True); plt.tight_layout()

    fn = os.path.join(output_folder, f"predicted_nodes_ellipse_n{n}.png")
    plt.savefig(fn, dpi=300); plt.close()

###############################################################################
# 6. Compute error_list & area_list (original print style)
###############################################################################
def compute_error_ellipse():
    analytical = math.pi * a * b
    levels     = [1,2,4,8,16]
    error_list = []
    area_list  = []

    print("\nComputing area by subdividing domain and calling GCNN per subcell for the ellipse:")
    for n in levels:
        pred_area = compute_h_refined_integral_ellipse(n)
        rel_error = abs(pred_area - analytical) / analytical
        area_list.append(pred_area)
        error_list.append(rel_error)

        print(f"  Subcells: {n}×{n}")
        print(f"    Predicted area : {pred_area:.16f}")
        print(f"    Analytical area: {analytical:.16f}")
        print(f"    Relative error : {rel_error:.16f}\n")

        save_subcell_nodes_plot_ellipse(n)

    # Plot Relative Error vs. Element Size (log-log)
    element_sizes = [2.0 / n for n in levels]
    plt.figure(figsize=(8,6))
    plt.plot(element_sizes, error_list, 'o-', color='b')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel("Element Size (2 / n) [log scale]")
    plt.ylabel("Relative Error [log scale]")
    plt.title("Relative Error vs. Element Size (Log-Log)")
    plt.grid(True, which='both', ls='--', alpha=0.7)
    fn = os.path.join(output_folder, "error_vs_element_size_ellipse.png")
    plt.savefig(fn, dpi=300); plt.close()

    # Plot Integral Area vs. Refinement Level
    plt.figure(figsize=(8,6))
    plt.plot(levels, area_list, 'o-', color='b', label='Predicted Integral Area')
    plt.axhline(analytical, color='r', linestyle='--', label='Analytical Area')
    plt.xlabel("Number of Subcells per Dimension (Refinement Level)")
    plt.ylabel("Integral Area")
    plt.title("Integral Area vs. Refinement Level")
    plt.legend(); plt.grid(True)
    fn = os.path.join(output_folder, "area_vs_refinement_ellipse.png")
    plt.savefig(fn, dpi=300); plt.close()

    return error_list, levels

###############################################################################
# 7. Main
###############################################################################
if __name__ == "__main__":
    compute_error_ellipse()
