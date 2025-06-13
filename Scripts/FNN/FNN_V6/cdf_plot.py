import numpy as np
import matplotlib.pyplot as plt
import os 

def extract_relative_errors(file_path, outlier_limit):
    errors = []
    with open(file_path, 'r') as file:
        for line in file:
            if "Relative Error" in line:
                try:
                    # Extract relative error values (they follow "Relative Error: ")
                    error_str = line.split("Relative Error:")[1].strip().replace('%', '')
                    error = float(error_str)
                    if error <= outlier_limit:  # Check if the error is within the acceptable range
                        errors.append(error)
                except IndexError:
                    # Handle any line that doesn't properly split (shouldn't occur if data is consistent)
                    print(f"Warning: Unable to process line: {line}")
    return errors

# Helper function to extract relative errors for GCNN model
def extract_gcnn_errors(file_path, error_limit=None):
    errors = []
    with open(file_path, 'r') as file:
        for line in file:
            if "Rel Error" in line:
                try:
                    error_str = line.split("Rel Error:")[1].split()[0].strip().replace('%', '')
                    error = float(error_str)
                    if error_limit is None or error <= error_limit:
                        errors.append(error)
                except IndexError:
                    print(f"Warning: Unable to process line: {line}")
                except ValueError:
                    print(f"Warning: Invalid relative error value in line: {line}")
    return errors

# Set an outlier limit, for example, 10% relative error
error_limit = 10

# Extract relative errors from the provided files, checking for outliers
fnn_errors = extract_relative_errors(r"C:\Git\RESULTS\Fnn\console_output_fnn.txt", error_limit)
cnn_errors = extract_relative_errors(r"C:\Git\RESULTS\Cnn\console_output_cnn.txt", error_limit)
hybrid_errors = extract_relative_errors(r"C:\Git\RESULTS\Hybrid\console_output_hybrid.txt", error_limit)
gcnn_errors = extract_gcnn_errors(r"C:\Git\RESULTS\Gcnn\console_output_gcnn.txt", error_limit)

# Print the number of extracted errors for each model
print(f"FNN Errors: {len(fnn_errors)}")
print(f"CNN Errors: {len(cnn_errors)}")
print(f"Hybrid Errors: {len(hybrid_errors)}")
print(f"GCNN Errors: {len(gcnn_errors)}")

# Function to compute CDF from a list of errors
def compute_cdf(errors):
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    return sorted_errors, cdf

# Compute CDF for each model
fnn_sorted, fnn_cdf = compute_cdf(fnn_errors)
cnn_sorted, cnn_cdf = compute_cdf(cnn_errors)
hybrid_sorted, hybrid_cdf = compute_cdf(hybrid_errors)
gcnn_sorted, gcnn_cdf = compute_cdf(gcnn_errors)

# Create the CDF plot (smooth curves)
plt.figure(figsize=(10, 6))
plt.plot(fnn_sorted, fnn_cdf, label="FNN", linestyle='-')
plt.plot(cnn_sorted, cnn_cdf, label="CNN", linestyle='-')
plt.plot(hybrid_sorted, hybrid_cdf, label="Hybrid", linestyle='-')
plt.plot(gcnn_sorted, gcnn_cdf, label="GCNN", linestyle='-')

# Labels and title
plt.xlabel('Relative Error (%)')
plt.ylabel('CDF')
plt.title('Cumulative Distribution of Relative Errors for 4 Models')
plt.legend()

# Display the plot
plt.grid(True)
output_file = os.path.join(os.getcwd(), 'relative_error_cdf_plot.png')
plt.savefig(output_file)