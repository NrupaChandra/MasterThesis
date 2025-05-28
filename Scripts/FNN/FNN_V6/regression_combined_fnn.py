#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from hrefinement_circle import compute_error_circle 
from hrefinement_ellipse import compute_error_ellipse

def linear_regression(element_sizes, error_list):
    # Convert data to log–log space.
    log_x = np.log(element_sizes)
    log_y = np.log(error_list)
    # Perform linear regression: log_y = slope * log_x + intercept.
    slope, intercept = np.polyfit(log_x, log_y, 1)
    # Compute regression line in the original space.
    fit_y = np.exp(slope * log_x + intercept)
    return slope, intercept, fit_y

def main():
    # Define output folder path.
    output_folder = r"C:\Git\MasterThesis\Scripts\FNN\FNN_V6\plt\Combined"

    # === Circle Data ===
    error_list_circle, refinement_levels_circle = compute_error_circle()
    error_list_circle = np.array(error_list_circle)
    refinement_levels_circle = np.array(refinement_levels_circle)
    # Compute element sizes: element size = 2 / n.
    element_sizes_circle = 2.0 / refinement_levels_circle
    slope_circle, intercept_circle, fit_y_circle = linear_regression(element_sizes_circle, error_list_circle)
    print(f"Circle: Fitted slope = {slope_circle:.6f}, intercept = {intercept_circle:.6f}")

    # === Ellipse Data ===
    error_list_ellipse, refinement_levels_ellipse = compute_error_ellipse()
    error_list_ellipse = np.array(error_list_ellipse)
    refinement_levels_ellipse = np.array(refinement_levels_ellipse)
    # Compute element sizes.
    element_sizes_ellipse = 2.0 / refinement_levels_ellipse
    slope_ellipse, intercept_ellipse, fit_y_ellipse = linear_regression(element_sizes_ellipse, error_list_ellipse)
    print(f"Ellipse: Fitted slope = {slope_ellipse:.6f}, intercept = {intercept_ellipse:.6f}")

    # === Plotting ===
    plt.figure(figsize=(10, 8))
    
    # Plot circle data and regression fit.
    plt.plot(element_sizes_circle, error_list_circle, 'o-', label=f'Circle Data (slope = {slope_circle:.6f})', color='blue')
    plt.plot(element_sizes_circle, fit_y_circle, '--', label='Circle Regression Fit', color='blue')
    
    # Plot ellipse data and regression fit.
    plt.plot(element_sizes_ellipse, error_list_ellipse, 's-', label=f'Ellipse Data (slope = {slope_ellipse:.6f})', color='green')
    plt.plot(element_sizes_ellipse, fit_y_ellipse, '--', label='Ellipse Regression Fit', color='green')
    
    # Set log scale for both axes.
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Element Size (log scale)")
    plt.ylabel("Relative Error (log scale)")
    plt.title("Log–Log Regression of Error vs. Element Size\nCircle and Ellipse")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)

    # Save and display the figure.
    plt.savefig(f"{output_folder}/error_regression_circle_ellipse_loglog.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
