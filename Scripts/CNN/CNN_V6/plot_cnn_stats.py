#!/usr/bin/env python3

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import LogFormatterMathtext

# ----------------------------
# 1. Style for Scientific Plots
# ----------------------------
plt.rc('font', family='serif', size=14)
plt.rc('axes', titlesize=16, labelsize=14, linewidth=1.5)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=12)

# ----------------------------
# 2. Path to Your Console File
# ----------------------------
# Update this path if your file is located elsewhere
file_path = r"C:\Git\RESULTS\Cnn\console_output.txt"

# ----------------------------
# 3. Parse the Console Output
# ----------------------------
# Adjusted regexes to match your actual console lines ("True Integral:", "Predicted Integral:")
true_pattern      = re.compile(r"True Integral:\s+([0-9.Ee+-]+)")
pred_pattern      = re.compile(r"Predicted Integral:\s+([0-9.Ee+-]+)")
abs_diff_pattern  = re.compile(r"Absolute Difference:\s+([0-9.Ee+-]+)")
rel_error_pattern = re.compile(r"Relative Error:\s+([0-9.]+)%")

true_integrals  = []
pred_integrals  = []
absolute_diffs  = []
relative_errors = []

with open(file_path, 'r') as f:
    lines = f.readlines()

for i in range(len(lines)):
    tm = true_pattern.search(lines[i])
    if tm:
        true_val = float(tm.group(1))

        # The next three lines should correspond to Predicted, Absolute Diff, and Relative Error
        pm = pred_pattern.search(lines[i + 1])
        am = abs_diff_pattern.search(lines[i + 2])
        rm = rel_error_pattern.search(lines[i + 3])

        if pm and am and rm:
            pred_val      = float(pm.group(1))
            abs_diff_val  = float(am.group(1))
            rel_error_val = float(rm.group(1))

            true_integrals.append(true_val)
            pred_integrals.append(pred_val)
            absolute_diffs.append(abs_diff_val)
            relative_errors.append(rel_error_val)

# Build DataFrame
df = pd.DataFrame({
    'True':     true_integrals,
    'Pred':     pred_integrals,
    'AbsDiff':  absolute_diffs,
    'RelErr':   relative_errors
})

# Save DataFrame for reference
df.to_csv('fnn_evaluation_results.csv', index=False)

# Compute summary statistics
median_abs = df['AbsDiff'].median()
median_rel = df['RelErr'].median()
Q1_rel = df['RelErr'].quantile(0.25)
Q3_rel = df['RelErr'].quantile(0.75)
IQR_rel = Q3_rel - Q1_rel
ub_rel = Q3_rel + 1.5 * IQR_rel

# ----------------------------
# 5. Plot 1: Scatter Plot (Predicted vs. True) with Zoom Inset
# ----------------------------
fig1, ax1 = plt.subplots(figsize=(6, 4))  # Unified size (width x height)

# Main scatter
ax1.scatter(
    df['True'], df['Pred'],
    s=20, alpha=0.6,
    color='tab:blue', edgecolors='k', linewidth=0.3
)

minv = min(df['True'].min(), df['Pred'].min())
maxv = max(df['True'].max(), df['Pred'].max())
margin = (maxv - minv) * 0.02
ax1.set_xlim(minv - margin, maxv + margin)
ax1.set_ylim(minv - margin, maxv + margin)

# Identity line
ax1.plot(
    [minv, maxv], [minv, maxv],
    color='dimgray',
    linestyle='--',
    linewidth=1.0,
    label='Ideal (y = x)'
)

ax1.set_xlabel('True Integral', fontsize=12)
ax1.set_ylabel('Predicted Integral', fontsize=12)
ax1.set_title('Predicted vs True Integrals - CNN', fontsize=12)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, linestyle='--', linewidth=0.5)

# Inset zoom (0–2.5)
axins = inset_axes(
    ax1, width="38%", height="38%", loc='lower right',
    bbox_to_anchor=(0.05, 0.05, 0.5, 0.5), bbox_transform=ax1.transAxes
)
axins.scatter(
    df['True'], df['Pred'],
    s=20, alpha=0.6,
    color='tab:blue', edgecolors='k', linewidth=0.3
)
axins.plot(
    [0, 2.5], [0, 2.5],
    color='dimgray',
    linestyle='--',
    linewidth=0.8
)
axins.set_xlim(0, 2.5)
axins.set_ylim(0, 2.5)
axins.set_xticks([0, 1, 2])
axins.set_yticks([0, 1, 2])
axins.grid(True, linestyle='--', linewidth=0.3)
axins.set_title('Zoom: 0 – 2.5', fontsize=10)

plt.tight_layout()
plt.savefig('scatter_pred_vs_true_improved_CNN.png', dpi=300)
plt.close()

# ----------------------------
# 8. Plot 4: Histogram of Relative Errors (Log‐Spaced Bins, 10^n Ticks, Blue Bars)
# ----------------------------
fig4, ax4 = plt.subplots(figsize=(6, 4))  # Match scatter plot size

min_nonzero = df.loc[df['RelErr'] > 0, 'RelErr'].min()
max_rel     = df['RelErr'].max()
bins_rel    = np.logspace(np.log10(min_nonzero), np.log10(max_rel), 50)

ax4.hist(
    df['RelErr'],
    bins=bins_rel,
    edgecolor='black',
    color='tab:blue',
    alpha=0.7
)

ax4.set_xscale('log')
ax4.axvline(median_rel, color='red', linestyle='--', linewidth=1.0, label=f'Median = {median_rel:.2f}%')
ax4.axvline(ub_rel, color='tab:blue', linestyle=':', linewidth=1.0, label=f'IQR Upper = {ub_rel:.2f}%')

ax4.set_xlabel('Relative Error (%)', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.set_title('Histogram of Relative Errors - CNN', fontsize=12)

formatter = LogFormatterMathtext(base=10)
ax4.xaxis.set_major_formatter(formatter)
ax4.set_xticks([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
ax4.tick_params(axis='x', labelsize=10)
ax4.tick_params(axis='y', labelsize=10)

ax4.legend(loc='upper right', fontsize=10)
ax4.grid(axis='y', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('hist_relative_error_log_bins_power10_blue_CNN.png', dpi=300)
plt.close()

print("All improved plots have been generated and saved:")
print("  • scatter_pred_vs_true_improved.png")
print("  • hist_relative_error_log_bins_power10_blue.png")
print("Parsed data also saved as fnn_evaluation_results.csv.")
