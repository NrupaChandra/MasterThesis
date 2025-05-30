#!/usr/bin/env python

import os
import time
import torch
import platform
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from multidataloader_fnn import MultiChunkDataset

# model loaders
from model_fnn    import load_ff_pipelines_model
from model_cnn    import load_shallow_cnn_model
from model_hybrid import load_shallow_cnn_model as load_hybrid_model
from model_gcnn   import load_gnn_model, NodalPreprocessor
from torch_geometric.nn import knn_graph

# --- Configuration ----------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

# Print device info
print(f"Using device: {device}")
if device.type == 'cuda':
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        gpu_name = "Unknown GPU"
    print(f" → GPU detected: {gpu_name}")
else:
    cpu_name = platform.processor() or platform.machine()
    cores    = os.cpu_count()
    print(f" → CPU detected: {cpu_name} ({cores} cores)")

data_dir = r"C:\Git\Data"
orders   = [1, 2, 3]

# Prepare storage for timings, including the extra "AlgoIM" curve:
times = {
    "FNN":    [],
    "CNN":    [],
    "Hybrid": [],
    "GCNN":   [],
    "AlgoIM": [2.269, 31.261, 2188.955]
}

# --- Nodal strings for Hybrid ----------------------------------------------
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
# --- Model weight paths ----------------------------------------------------
paths = {
    "FNN":    r"C:\Git\MasterThesis\Models\FNN\FNN_model_v6\fnn_model_weights_v6.pth",
    "CNN":    r"C:\Git\MasterThesis\Models\CNN\CNN_V6\cnn_model_weights_v5.0.pth",
    "Hybrid": r"C:\Git\MasterThesis\Models\Hybrid\Hybrid_V5\fnn_model_weights_v4.pth",
    "GCNN":   r"C:\Git\MasterThesis\Models\GCN\GCN_v3\gcnn_model_weights_v3.pth"
}

# --- Load, time, & benchmark -----------------------------------------------
for name in ( "Hybrid", "GCNN"):
    print(f"\n>>> Loading {name} model")

    # 1) Build empty skeleton
    if name == "FNN":
        model = load_ff_pipelines_model(weights_path=None)

    elif name == "CNN":
        model = load_shallow_cnn_model(weights_path=None)

    elif name == "Hybrid":
        model = load_hybrid_model(
            weights_path=None,
            node_x_str=node_x_str,
            node_y_str=node_y_str
        )

    else:  # GCNN
        model = load_gnn_model(
            in_channels=3,
            hidden_channels=64,
            num_layers=5,
            dropout_rate=0.0,
            device=device
        )

    # 2) Load weights
    sd = torch.load(paths[name], map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)

    # 3) Move & 4) Eval
    model.to(device)
    model.eval()

    # --- Inference timing --------------------------------------------------
    with torch.no_grad():
        for p in orders:
            idx_file = os.path.join(data_dir, f"preprocessed_chunks_{p}_10k", "index.txt")
            ds = MultiChunkDataset(index_file=idx_file, base_dir=data_dir)
            dl = DataLoader(ds, batch_size=1, shuffle=False)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            for exp_x, exp_y, coeff, _ in dl:
                exp_x, exp_y, coeff = exp_x.to(device), exp_y.to(device), coeff.to(device)

                if name == "GCNN":
                    # rebuild graph so it matches this sample
                    pos        = torch.stack([exp_x.flatten(), exp_y.flatten()], dim=1)
                    edge_index = knn_graph(pos, k=4, loop=False).to(device)
                    x_in = torch.cat([
                        exp_x.flatten()[:, None],
                        exp_y.flatten()[:, None],
                        coeff.flatten()[:, None]
                    ], dim=1)
                    _ = model(x_in, edge_index)
                else:
                    _ = model(exp_x, exp_y, coeff)

                if device.type == "cuda":
                    torch.cuda.synchronize()

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()

            times[name].append(t1 - t0)
            print(f"  {name} — order {p}: {t1 - t0:.3f} s")

# --- Plot & Save (log scale, elongated x-axis) ------------------------------
fig, ax = plt.subplots(figsize=(10,4))

# 1) Plot each curve with thin lines & markers
for name, vals in times.items():
    ax.plot(
        orders, vals,
        marker='o',
        markersize=6,
        linewidth=1.0,
        label=name
    )

# 2) Vertical marker at polynomial order 2
ax.axvline(2, color='gray', linestyle='-', linewidth=1.0, alpha=0.7, zorder=0)

# 3) Axes limits & ticks
ax.set_xticks(orders)
ax.set_xlim(0.8, 3.2)   # stretch from 0.8 to 3.2
ax.set_yscale('log')

# 4) Grids: solid major, dashed minor
ax.minorticks_on()
ax.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.8)
ax.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.6)

# 5) Spine thickness
for spine in ax.spines.values():
    spine.set_linewidth(1.0)

# 6) Labels, title, legend
ax.set_xlabel("Polynomial order")
ax.set_ylabel("Execution Time (s)")
ax.set_title("ML vs AlgoIM Inference Time by Polynomial Order")
ax.legend(frameon=True, framealpha=1, edgecolor='gray', fontsize='small')

plt.tight_layout()

# 7) Save and show
out_path = os.path.join(os.getcwd(), "inference_performance_vs_order_final.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {out_path}")
plt.show()
