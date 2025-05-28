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
# Circle parameters
###############################################################################
radius = 0.4
a, b = radius, radius    # same for circle
C_x, C_y = 0.0, 0.0      # center at origin
angle = 0.0              # no rotation

###############################################################################
# 1. Load the GCNN Model & Build Graph
###############################################################################
device        = torch.device('cpu')
model_path    = r"C:\Git\MasterThesis\Models\GCN\GCN_v3\gcnn_model_weights_v3.pth"
output_folder = r"C:\Git\MasterThesis\Scripts\GCNN\plt\circle"
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

# build preprocessor and static graph
preprocessor = NodalPreprocessor(node_x_str, node_y_str).to(device)
pos = torch.stack([preprocessor.X, preprocessor.Y], dim=1).to(device)
edge_index = knn_graph(pos, k=4, loop=False).to(device)

# instantiate & load model
model = load_gnn_model(in_channels=3, hidden_channels=64, num_layers=5,
                       dropout_rate=0.0, device=device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

###############################################################################
# 2. Build the Circle Polynomial in a Subcell (normalize by h^2)
###############################################################################
def make_subcell_circle_polynomial(ox, oy, n):
    h = 1.0 / n
    # f_phys(x,y) = -1 + (1/radius^2)*(x^2+y^2)  scaled so zero contour at circle
    # here radius=0.4 => factor = 1/(0.4^2)=6.25
    factor = 1.0/(radius*radius)
    # Expand f_phys(ox + hX, oy + hY)/h^2
    A0 = factor*(ox*ox + oy*oy) - 1.0
    AX = 2*factor*ox*h
    AY = 2*factor*oy*h
    AX2 = factor*(h*h)
    AY2 = AX2

    c0  = A0/(h*h)
    cX  = AX/(h*h)
    cY  = AY/(h*h)
    cX2 = AX2/(h*h)
    cY2 = AY2/(h*h)

    exps_x = torch.tensor([[0,1,2,0,0]], dtype=torch.float32, device=device)
    exps_y = torch.tensor([[0,0,0,1,2]], dtype=torch.float32, device=device)
    coeffs = torch.tensor([[c0, cX, cX2, cY, cY2]],
                          dtype=torch.float32, device=device)
    return exps_x, exps_y, coeffs

###############################################################################
# 3. Circle "inside" checker
###############################################################################
def is_inside_circle(x, y):
    return (x*x + y*y) <= radius*radius

###############################################################################
# 4. Subcell-based Integration with full/empty check + GCNN fallback
###############################################################################
def compute_h_refined_integral(n):
    h   = 1.0 / n
    jac = h*h
    h2  = 0.5*h
    centers = np.linspace(-1 + h, 1 - h, n)
    total = 0.0

    for ox in centers:
        for oy in centers:
            # full/empty test (only if n>1)
            if n > 1:
                corners = [(ox + dx*h, oy + dy*h) for dx in (-1,1) for dy in (-1,1)]
                flg = [is_inside_circle(xc, yc) for xc,yc in corners]
                center_flg = is_inside_circle(ox, oy)
                if all(flg):
                    total += jac * 4.0
                    continue
                if not any(flg) and not center_flg:
                    continue

            # partial ⇒ run GCNN
            ex, ey, cf = make_subcell_circle_polynomial(ox, oy, n)
            with torch.no_grad():
                nodal  = preprocessor(ex, ey, cf)[0]             # [P]
                feats  = torch.cat([pos, nodal.unsqueeze(1)],1)  # [P,3]
                shifts, weights = model(feats, edge_index)       # [P,2], [P]

            # map to physical: base + uniform X/Y + half-shift
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
# 5. (Optional) Plot partial-cell nodes exactly like CNN circle version
###############################################################################
def save_subcell_nodes_plot(n, filename):
    h, h2 = 1.0/n, 0.5/n
    centers = np.linspace(-1 + h, 1 - h, n)
    Xs, Ys, Ws = [], [], []

    plt.figure(figsize=(8,8))
    for ox in centers:
        for oy in centers:
            if n == 1:
                case = 'partial'
            else:
                corners = [(ox+dx*h, oy+dy*h) for dx in(-1,1) for dy in(-1,1)]
                flg = [is_inside_circle(xc,yc) for xc,yc in corners]
                cfl = is_inside_circle(ox, oy)
                if all(flg):
                    case = 'inside'
                elif not any(flg) and not cfl:
                    case = 'outside'
                else:
                    case = 'partial'

            if case=='inside':
                plt.gca().add_patch(plt.Rectangle((ox-h,oy-h),2*h,2*h,
                    facecolor='lightgreen', edgecolor='blue',
                    linestyle='--', alpha=0.5))
            elif case=='outside':
                plt.gca().add_patch(plt.Rectangle((ox-h,oy-h),2*h,2*h,
                    facecolor='lightgray', edgecolor='blue',
                    linestyle='--', alpha=0.3))
            else:
                ex, ey, cf = make_subcell_circle_polynomial(ox, oy, n)
                with torch.no_grad():
                    nodal = preprocessor(ex, ey, cf)[0]
                    feats = torch.cat([pos, nodal.unsqueeze(1)],1)
                    shifts, weights = model(feats, edge_index)

                pts = torch.stack([
                    ox + h*pos[:,0] + h2*shifts[:,0],
                    oy + h*pos[:,1] + h2*shifts[:,1]
                ],1).cpu().numpy()
                Xs.append(pts[:,0]); Ys.append(pts[:,1]); Ws.append(weights.cpu().numpy())

    if Xs:
        X = np.concatenate(Xs); Y = np.concatenate(Ys); W = np.concatenate(Ws)
        sc = plt.scatter(X, Y, c=W, cmap='viridis', s=10, edgecolors='k')
        plt.colorbar(sc, label="Predicted Weight")

    # true boundary
    θ = np.linspace(0,2*np.pi,200)
    x_circ = radius*np.cos(θ); y_circ = radius*np.sin(θ)
    plt.plot(x_circ, y_circ, 'r-', linewidth=2, label='Circle Boundary')

    for i in range(n+1):
        c = -1 + 2*i*h
        plt.axvline(c, ls='--', color='blue', linewidth=0.5)
        plt.axhline(c, ls='--', color='blue', linewidth=0.5)

    plt.title(f"Subcell-based Predicted Nodes (n={n} per dim)")
    plt.xlabel("X"); plt.ylabel("Y")
    plt.gca().set_aspect('equal','box')
    plt.xlim(-1,1); plt.ylim(-1,1)
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

###############################################################################
# 6. Compute error_list & area_list (identical print style)
###############################################################################
def compute_error_circle():
    analytical_area = math.pi * (radius**2)
    levels = [1,2,4,8,16]
    area_list, error_list = [], []

    print("\nComputing area by subdividing domain and calling GCNN per subcell for the circle:")
    for n in levels:
        A = compute_h_refined_integral(n)
        err = abs(A - analytical_area) / analytical_area
        area_list.append(A); error_list.append(err)

        print(f"  Subcells: {n}x{n}")
        print(f"    Predicted area : {A:.16f}")
        print(f"    Analytical area: {analytical_area:.16f}")
        print(f"    Relative error : {err:.16f}\n")

        png = os.path.join(output_folder, f"predicted_nodes_circle_n{n}.png")
        save_subcell_nodes_plot(n, png)
        print(f"Aggregate subcell plot saved as '{png}'.")

    # error vs element size (log-log)
    element_sizes = [2.0/n for n in levels]
    plt.figure(figsize=(8,6))
    plt.plot(element_sizes, error_list, 'o-', color='b')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel("Element Size (2 / n) [log scale]")
    plt.ylabel("Relative Error [log scale]")
    plt.title("Relative Error vs. Element Size (Log-Log) for Circle")
    plt.grid(True, which='both', ls='--', alpha=0.7)
    fn = os.path.join(output_folder, "error_vs_element_size_circle.png")
    plt.savefig(fn, dpi=300); plt.close()
    print(f"Relative error vs. element size plot saved as '{fn}'.")

    # area vs refinement
    plt.figure(figsize=(8,6))
    plt.plot(levels, area_list, 'o-', color='b', label='Predicted Integral Area')
    plt.axhline(analytical_area, color='r', linestyle='--', label='Analytical Area')
    plt.xlabel("Number of Subcells per Dimension (Refinement Level)")
    plt.ylabel("Integral Area")
    plt.title("Integral Area vs. Refinement Level for Circle")
    plt.legend(); plt.grid(True)
    fn = os.path.join(output_folder, "area_vs_refinement_circle.png")
    plt.savefig(fn, dpi=300); plt.close()
    print(f"Integral area plot saved as '{fn}'.")

    return error_list, levels

###############################################################################
# 7. Main
###############################################################################
if __name__ == "__main__":
    compute_error_circle()
