# main_script.py
import numpy as np
from reader import merged_data_1st_order  
from reader import merged_data_2nd_order 

# Function to prepare the feature matrix
def prepare_feature_matrix(data):
    feature_matrix = []
    for entry in data:
        # Combine exp_x, exp_y, and coeff into a single feature vector
        feature_vector = entry['exp_x'] + entry['exp_y'] + entry['coeff']
        feature_matrix.append(feature_vector)
    
    return np.array(feature_matrix)

# Prepare feature matrix for 1st order data
feature_matrix_1st_order = prepare_feature_matrix(merged_data_1st_order)
feature_matrix_2nd_order = prepare_feature_matrix(merged_data_2nd_order)

# Print feature matrix 
print("1st Order Merged Data Sample:", feature_matrix_1st_order[:1])  # Show first two entries for preview
print("2nd Order Merged Data Sample:", feature_matrix_2nd_order[:1])  # Show first two entries for preview
