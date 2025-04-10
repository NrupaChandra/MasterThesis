#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from hrefinement_circle import compute_error_circle 
from hrefinement_ellipse import compute_error_ellipse
from hrefinement_triangle import compute_error_triangle

def main():
    output_folder = r"C:\Git\MasterThesis\Scripts\CNN\CNN_V1\plt\Triangle"
    
    # Get error_list and refinement_levels from compute_area module
    error_list, refinement_levels = compute_error_triangle()
    error_list = np.array(error_list)
    refinement_levels = np.array(refinement_levels)
    
    # Compute element sizes: element size = 2 / n.
    element_sizes = 2.0 / refinement_levels

    # Convert to log-log space.
    log_x = np.log(element_sizes)
    log_y = np.log(error_list)

    # Perform linear regression in log space: log_y ~ slope * log_x + intercept
    slope, intercept = np.polyfit(log_x, log_y, 1)
    print(f"Fitted slope in log-log space = {slope:.6f}")
    print(f"Fitted intercept in log-log space = {intercept:.6f}")

    # Generate regression line for plotting.
    fit_log_y = slope * log_x + intercept
    fit_y = np.exp(fit_log_y)

    # Plot original data and regression on a log-log plot.
    plt.figure(figsize=(8,6))
    plt.plot(element_sizes, error_list, 'o-', label='Data', color='blue')
    plt.plot(element_sizes, fit_y, '--', label='Regression Fit', color='red')

    plt.xscale('log')
    plt.yscale('log')
    # If the original plot had the y-axis inverted, uncomment the following line:
    # plt.gca().invert_yaxis()
    plt.xlabel("Element Size (log scale)")
    plt.ylabel("Relative Error (log scale)")
    plt.title(f"Log-Log Regression of Error vs. Element Size - triangle, slope = {slope:.6f}")

    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)

   

    plt.savefig(f"{output_folder}/error_regression_loglog.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
