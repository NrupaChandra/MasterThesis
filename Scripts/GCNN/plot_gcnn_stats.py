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
file_path = r"C:\Git\RESULTS\Gcnn\results_log.txt"

# ----------------------------
# 3. Parse the Console Output
# ----------------------------
# Updated regexes to match the actual console lines from your log:
true_pattern       = re.compile(r"^\s*True Int:\s*([0-9.Ee+\-]+)\s*$")
pred_pattern       = re.compile(r"^\s*Pred Int:\s*([0-9.Ee+\-]+)\s*$")
rel_error_pattern  = re.compile(r"^\s*Rel Error:\s*([0-9.]+)%\s*$")
mse_pattern        = re.compile(r"^\s*Sample MSE:\s*([0-9.Ee+\-]+)\s*$")

true_integrals   = []
pred_integrals   = []
absolute_diffs   = []   # we'll compute abs diff = sqrt(sample_mse)
relative_errors  = []

with open(file_path, 'r') as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    # 1) Look for a "True Int:" line
    m_true = true_pattern.match(lines[i])
    if not m_true:
        i += 1
        continue

    # If we found a True Int, try to capture the next three lines in order:
    #   - Pred Int
    #   - Rel Error
    #   - Sample MSE
    if i + 3 >= len(lines):
        raise RuntimeError("Incomplete block after True Int at line %d" % i)

    m_pred = pred_pattern.match(lines[i + 1])
    m_rel  = rel_error_pattern.match(lines[i + 2])
    m_mse  = mse_pattern.match(lines[i + 3])

    if not (m_pred and m_rel and m_mse):
        # If any of those three lines don’t match, skip ahead and keep scanning.
        i += 1
        continue

    # Parse the numbers out of the regex groups
    true_val      = float(m_true.group(1))
    pred_val      = float(m_pred.group(1))
    rel_err_pct   = float(m_rel.group(1))
    sample_mse    = float(m_mse.group(1))

    # Compute absolute difference as sqrt(MSE)
    abs_diff_val = np.sqrt(sample_mse)

    true_integrals.append(true_val)
    pred_integrals.append(pred_val)
    absolute_diffs.append(abs_diff_val)
    relative_errors.append(rel_err_pct)

    # Jump past these four lines for the next iteration
    i += 4

# Build DataFrame
df = pd.DataFrame({
    'True':     true_integrals,
    'Pred':     pred_integrals,
    'AbsDiff':  absolute_diffs,
    'RelErr':   relative_errors
})

# If df is empty, warn and exit gracefully
if df.empty:
    raise RuntimeError("Parsed DataFrame is empty—no matching lines were found in results_log.txt.")

# Save DataFrame for reference
df.to_csv('fnn_evaluation_results.csv', index=False)

# Compute summary statistics
median_abs = df['AbsDiff'].median()
median_rel = df['RelErr'].median()
Q1_rel     = df['RelErr'].quantile(0.25)
Q3_rel     = df['RelErr'].quantile(0.75)
IQR_rel    = Q3_rel - Q1_rel
ub_rel     = Q3_rel + 1.5 * IQR_rel

# ----------------------------
# 5. Plot 1: Scatter Plot (Predicted vs. True) with Zoom Inset
# ----------------------------
fig1, ax1 = plt.subplots(figsize=(6, 6))

# Main scatter
ax1.scatter(
    df['True'], df['Pred'],
    s=20, alpha=0.6,
    color='tab:blue', edgecolors='k', linewidth=0.3
)

minv   = min(df['True'].min(), df['Pred'].min())
maxv   = max(df['True'].max(), df['Pred'].max())
margin = (maxv - minv) * 0.02
ax1.set_xlim(minv - margin, maxv + margin)
ax1.set_ylim(minv - margin, maxv + margin)

# Draw a thinner, gray dashed identity line (y = x)
ax1.plot(
    [minv, maxv], [minv, maxv],
    color='dimgray',
    linestyle='--',
    linewidth=1.0,
    label='Ideal (y = x)'
)

ax1.set_xlabel('True Integral')
ax1.set_ylabel('Predicted Integral')
ax1.set_title('Predicted vs True Integrals - GCNN')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, linestyle='--', linewidth=0.5)

# Inset zoom (0–2.5)
axins = inset_axes(
    ax1, width="40%", height="40%", loc='lower right',
    bbox_to_anchor=(0.05, 0.05, 0.5, 0.5), bbox_transform=ax1.transAxes
)
axins.scatter(
    df['True'], df['Pred'],
    s=20, alpha=0.6,
    color='tab:blue', edgecolors='k', linewidth=0.3
)
# Thinner gray dashed inset line
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
plt.savefig('scatter_pred_vs_true_improved_gcnn.png', dpi=300)
plt.close()

# ----------------------------
# 8. Plot 4: Histogram of Relative Errors (Log-Spaced Bins, 10^n Ticks, Blue Bars)
# ----------------------------
fig4, ax4 = plt.subplots(figsize=(6, 4))

# Compute smallest nonzero relative error and the maximum
min_nonzero = df.loc[df['RelErr'] > 0, 'RelErr'].min()
max_rel     = df['RelErr'].max()

# Build 50 log-spaced bins between those endpoints
bins_rel = np.logspace(np.log10(min_nonzero), np.log10(max_rel), 50)

# Plot histogram in blue
ax4.hist(
    df['RelErr'],
    bins=bins_rel,
    edgecolor='black',
    color='tab:blue',
    alpha=0.7
)

# Switch x-axis to log scale
ax4.set_xscale('log')

# Thin vertical lines: median (red dashed) and IQR upper (mid-blue dotted)
ax4.axvline(
    median_rel,
    color='red',
    linestyle='--',
    linewidth=1.0,
    label=f'Median = {median_rel:.2f}%'
)
ax4.axvline(
    ub_rel,
    color='tab:blue',
    linestyle=':',
    linewidth=1.0,
    label=f'IQR Upper = {ub_rel:.2f}%'
)

# Axis labels & title
ax4.set_xlabel('Relative Error (%)', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.set_title('Histogram of Relative Errors - GCNN', fontsize=12)

# Use math-text formatter so x-ticks read as 10^n
formatter = LogFormatterMathtext(base=10)
ax4.xaxis.set_major_formatter(formatter)

# Place ticks at explicit decades
ax4.set_xticks([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])

# Reduce tick-label sizes
ax4.tick_params(axis='x', labelsize=10)
ax4.tick_params(axis='y', labelsize=10)

# Legend + grid
ax4.legend(loc='upper right', fontsize=10)
ax4.grid(axis='y', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('hist_relative_error_log_bins_power10_blue_gcnn.png', dpi=300)
plt.close()

print("All improved plots have been generated and saved:")
print("  • scatter_pred_vs_true_improved.png")
print("  • hist_relative_error_log_bins_power10_blue.png")
print("Parsed data also saved as fnn_evaluation_results.csv.")
