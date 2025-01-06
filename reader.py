def read_level_set_file(file_path):
    """Reads a level set file and parses its data (polynomial info)."""
    data = {}
    with open(file_path, 'r') as file:
        # Skip the header line
        next(file)
        for line in file:
            # Skip empty lines
            if not line.strip():
                continue
            # Split the line into components by the semicolon delimiter
            parts = line.strip().split(';')
            if len(parts) == 5:  # Ensure we have ID, exp_x, exp_y, and coeff
                number = int(parts[0])  # Parse the number
                id_ = parts[1]          # Unique ID
                exp_x = list(map(int, parts[2].split(',')))  # Exponents of x
                exp_y = list(map(int, parts[3].split(',')))  # Exponents of y
                coeff = list(map(float, parts[4].split(',')))  # Coefficients
                # Store polynomial data by 'number' for later merging
                data[number] = {
                    'id': id_,
                    'exp_x': exp_x,
                    'exp_y': exp_y,
                    'coeff': coeff
                }
    return data

def read_nodes_weights_file(file_path):
    """Reads the file containing nodes and weights."""
    data = {}
    with open(file_path, 'r') as file:
        # Skip the header line
        next(file)
        for line in file:
            # Skip empty lines
            if not line.strip():
                continue
            # Split the line into components by the semicolon delimiter
            parts = line.strip().split(';')
            if len(parts) == 5:  # Ensure there are nodes_x, nodes_y, and weights
                number = int(parts[0])  # Parse the number
                id_ = parts[1]          # Unique ID
                nodes_x = list(map(float, parts[2].split(',')))  # Nodes in x
                nodes_y = list(map(float, parts[3].split(',')))  # Nodes in y
                weights = list(map(float, parts[4].split(',')))  # Weights
                # Store nodes and weights data by 'number' for later merging
                data[number] = {
                    'id': id_,
                    'nodes_x': nodes_x,
                    'nodes_y': nodes_y,
                    'weights': weights
                }
    return data

# Correct file paths
file_1st_order = r'C:\Git\QuadRuleGeneration\Bernstein_p1_data.txt'
file_2nd_order = r'C:\Git\QuadRuleGeneration\Bernstein_p2_data.txt'
file_nodes_weights_1D = r'C:\Git\QuadRuleGeneration\Bernstein_p1_output.txt'
file_nodes_weights_2D = r'C:\Git\QuadRuleGeneration\Bernstein_p1_output.txt'

# Read data from level set and node/weight files
data_1st_order = read_level_set_file(file_1st_order)
data_2nd_order = read_level_set_file(file_2nd_order)
nodes_weights_1D = read_nodes_weights_file(file_nodes_weights_1D)
nodes_weights_2D = read_nodes_weights_file(file_nodes_weights_2D)

"""Debugging: Check if the files were read correctly
print("1st Order Polynomial Data Sample:", list(data_1st_order.items())[:2])
print("2nd Order Polynomial Data Sample:", list(data_2nd_order.items())[:2])
print("1D Node/Weight Data Sample:", list(nodes_weights_1D.items())[:2])
print("2D Node/Weight Data Sample:", list(nodes_weights_2D.items())[:2])"""

# Merge polynomial data with nodes and weights
def merge_data(polynomial_data, node_weight_data):
    """Merge polynomial data with corresponding nodes and weights using 'number'."""
    merged_data = []
    for number, poly_data in polynomial_data.items():
        if number in node_weight_data:
            node_data = node_weight_data[number]
            merged_data.append({
                **poly_data,  # Include polynomial features (exp_x, exp_y, coeff)
                'nodes_x': node_data['nodes_x'],
                'nodes_y': node_data['nodes_y'],
                'weights': node_data['weights']
            })
    return merged_data

# Merge 1st and 2nd order data
merged_data_1st_order = merge_data(data_1st_order, nodes_weights_1D)
merged_data_2nd_order = merge_data(data_2nd_order, nodes_weights_2D)

# Check if the merge is successful
print("1st Order Merged Data Sample:", merged_data_1st_order[:1])  # Show first two entries for preview
#print("2nd Order Merged Data Sample:", merged_data_2nd_order[:2])  # Show first two entries for preview
