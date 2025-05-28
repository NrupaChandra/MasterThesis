#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib.util

# === CONFIGURATION ===
MODEL_PATHS = {
    'GCNN':   r'C:\Git\MasterThesis\Scripts\GCNN',
    'FNN':    r'C:\Git\MasterThesis\Scripts\FNN\FNN_V6',
    'CNN':    r'C:\Git\MasterThesis\Scripts\CNN\CNN_V6',
    'Hybrid': r'C:\Git\MasterThesis\Scripts\CNN_hybrid\Hybrid_V5'
}

# === GLOBAL STYLE SETTINGS ===
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'grid.linestyle': ':',
    'grid.linewidth': 0.8,
    'grid.alpha': 0.6,
})


def import_module_from_path(module_name, base_path):
    file_path = os.path.join(base_path, f"{module_name}.py")
    sys.path.insert(0, base_path)
    try:
        spec = importlib.util.spec_from_file_location(f"{base_path}_{module_name}", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def linear_regression(x, y):
    log_x, log_y = np.log(x), np.log(y)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    fit = np.exp(slope * log_x + intercept)
    return slope, fit


def plot_refinement(data_dict, title, output_folder, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = ['o', 's', '^', 'd']
    colors  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for (model, (h, err, slope, fit, levels)), m, c in zip(data_dict.items(), markers, colors):
        ax.plot(h, err, marker=m, mfc='white', mec=c, ms=8, linestyle='-', lw=1.5,
                label=f"{model} (s={slope:.2f})", color=c)
        ax.plot(h, fit, linestyle='--', lw=1.2, color=c)
        for xi, yi, lvl in zip(h, err, levels):
            ax.text(xi, yi, str(lvl), fontsize=9, ha='center', va='bottom')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Element Size (h)')
    ax.set_ylabel('Relative Error')
    ax.set_title(title)
    ax.grid(True, which='both')
    ax.legend(frameon=False)

    os.makedirs(output_folder, exist_ok=True)
    out = os.path.join(output_folder, filename)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.show()


def main():
    out_dir = r'C:\Git\MasterThesis\Scripts\Combined_Plots'
    circle_data, ellipse_data = {}, {}

    for model, path in MODEL_PATHS.items():
        cmod = import_module_from_path('hrefinement_circle', path)
        emod = import_module_from_path('hrefinement_ellipse', path)

        ec, lc = cmod.compute_error_circle()
        he = 2.0 / np.array(lc)
        ec = np.array(ec)
        sc, fitc = linear_regression(he, ec)
        circle_data[model] = (he, ec, sc, fitc, lc)

        ee, le = emod.compute_error_ellipse()
        he2 = 2.0 / np.array(le)
        ee = np.array(ee)
        se, fite = linear_regression(he2, ee)
        ellipse_data[model] = (he2, ee, se, fite, le)

    plot_refinement(circle_data,  'Circle $h$-Refinement Convergence',  out_dir, 'circle_refine.png')
    plot_refinement(ellipse_data,'Ellipse $h$-Refinement Convergence', out_dir, 'ellipse_refine.png')

if __name__ == '__main__':
    main()
