# main_script.py
import numpy as np
from reader import merged_data_1st_order  , merged_data_2nd_order 

# Function to prepare the feature matrix
def prepare_feature_matrix(data):
    feature_matrix = []
    for entry in data:
        # Combine exp_x, exp_y, and coeff into a single feature vector
        feature_vector = entry['exp_x'] + entry['exp_y'] + entry['coeff']
        feature_matrix.append(feature_vector)
    
    return np.array(feature_matrix)

# Function to prepare the target matrix
def prepare_target_matrix(data):
    target_matrix = []
    
    # Determine the maximum length for nodes_x, nodes_y, and weights
    max_length = max([len(entry['nodes_x']) for entry in data] +
                     [len(entry['nodes_y']) for entry in data] +
                     [len(entry['weights']) for entry in data])
    
    for entry in data:
        # Pad the lists to ensure they are all the same length

        '''Padding: The function now calculates the max_length of the lists (nodes_x, nodes_y, and weights)
           across all entries in the dataset. If a list is shorter than max_length, it is padded with zeros.
           Consistent Shape: All lists are padded to ensure they have the same length, avoiding the 
           inhomogeneous shape error.'''
        
        padded_nodes_x = entry['nodes_x'] + [0] * (max_length - len(entry['nodes_x']))
        padded_nodes_y = entry['nodes_y'] + [0] * (max_length - len(entry['nodes_y']))
        padded_weights = entry['weights'] + [0] * (max_length - len(entry['weights']))
        
        # Combine the padded lists into a single target vector
        target_vector = padded_nodes_x + padded_nodes_y + padded_weights
        target_matrix.append(target_vector)
    
    return np.array(target_matrix)


# Prepare feature matrix 
feature_matrix_1st_order = prepare_feature_matrix(merged_data_1st_order)
feature_matrix_2nd_order = prepare_feature_matrix(merged_data_2nd_order)

# Prepare target matrix 
target_matrix_1st_order = prepare_target_matrix(merged_data_1st_order)
target_matrix_2nd_order = prepare_target_matrix(merged_data_2nd_order)

# Print feature matrix 
print("1st Order Merged Data Sample:", feature_matrix_1st_order[:1])  # Show first two entries for preview
print("2nd Order Merged Data Sample:", feature_matrix_2nd_order[:1])  # Show first two entries for preview

# Print target matrix 
print("1st Order Merged Data Sample:", target_matrix_1st_order[:2])  # Show first two entries for preview
print("2nd Order Merged Data Sample:", target_matrix_2nd_order[:2])  # Show first two entries for preview