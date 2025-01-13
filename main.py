import numpy as np
from reader import merged_data_1st_order, merged_data_2nd_order
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Function to prepare the feature matrix
def prepare_feature_matrix(data):
    feature_matrix = []
    for entry in data:
        feature_vector = entry['exp_x'] + entry['exp_y'] + entry['coeff']
        feature_matrix.append(feature_vector)
    return np.array(feature_matrix)

# Function to prepare the target matrix with padding
def prepare_target_matrix(data):
    target_matrix = []
    max_length = max([len(entry['nodes_x']) for entry in data] +
                     [len(entry['nodes_y']) for entry in data] +
                     [len(entry['weights']) for entry in data])
    
    for entry in data:
        padded_nodes_x = entry['nodes_x'] + [0] * (max_length - len(entry['nodes_x']))
        padded_nodes_y = entry['nodes_y'] + [0] * (max_length - len(entry['nodes_y']))
        padded_weights = entry['weights'] + [0] * (max_length - len(entry['weights']))
        target_vector = padded_nodes_x + padded_nodes_y + padded_weights
        target_matrix.append(target_vector)
    
    return np.array(target_matrix)

# Prepare feature and target matrices for 1st order data
feature_matrix_1st_order = prepare_feature_matrix(merged_data_1st_order)
target_matrix_1st_order = prepare_target_matrix(merged_data_1st_order)

# Prepare feature and target matrices for 2nd order data
feature_matrix_2nd_order = prepare_feature_matrix(merged_data_2nd_order)
target_matrix_2nd_order = prepare_target_matrix(merged_data_2nd_order)

# Split data into training and testing sets (80% for training, 20% for testing)
X_train_1st, X_test_1st, y_train_1st, y_test_1st = train_test_split(feature_matrix_1st_order, target_matrix_1st_order, test_size=0.2, random_state=42)
X_train_2nd, X_test_2nd, y_train_2nd, y_test_2nd = train_test_split(feature_matrix_2nd_order, target_matrix_2nd_order, test_size=0.2, random_state=42)

# Initialize a simple linear regression model
model = LinearRegression()

# Train the model on the 1st order data
model.fit(X_train_1st, y_train_1st)

# Predict on the test data for 1st order
y_pred_1st = model.predict(X_test_1st)

# Evaluate the model performance using Mean Squared Error (MSE) for 1st order data
mse_1st = mean_squared_error(y_test_1st, y_pred_1st)
print("Mean Squared Error on 1st Order Test Data:", mse_1st)

# Train the model on the 2nd order data
model.fit(X_train_2nd, y_train_2nd)

# Predict on the test data for 2nd order
y_pred_2nd = model.predict(X_test_2nd)

# Evaluate the model performance using Mean Squared Error (MSE) for 2nd order data
mse_2nd = mean_squared_error(y_test_2nd, y_pred_2nd)
print("Mean Squared Error on 2nd Order Test Data:", mse_2nd)

# Function to save predictions to a text file in the specified format
def save_predictions_to_text_file(predictions, data, file_path):
    with open(file_path, 'w') as file:
        # Write the header
        file.write("number;id;nodes_x;nodes_y;weights\n")
        
        # Write the predictions and corresponding ids
        for idx, (entry, prediction) in enumerate(zip(data, predictions)):
            # Get the id
            entry_id = entry['id']
            
            # Get the original number of nodes_x, nodes_y, and weights for this entry
            num_nodes_x = len(entry['nodes_x'])  # Original number of nodes_x
            num_nodes_y = len(entry['nodes_y'])  # Original number of nodes_y
            num_weights = len(entry['weights'])  # Original number of weights
            
            # Extract the predicted nodes_x, nodes_y, and weights
            predicted_nodes_x = prediction[:num_nodes_x]  # Extract corresponding nodes_x
            predicted_nodes_y = prediction[num_nodes_x:num_nodes_x + num_nodes_y]  # Extract corresponding nodes_y
            predicted_weights = prediction[num_nodes_x + num_nodes_y:num_nodes_x + num_nodes_y + num_weights]  # Extract corresponding weights
            
            # Format the data as comma-separated values
            nodes_x_str = ",".join(map(str, predicted_nodes_x))
            nodes_y_str = ",".join(map(str, predicted_nodes_y))
            weights_str = ",".join(map(str, predicted_weights))
            
            # Format the entire row
            formatted_data = f"{idx};{entry_id};{nodes_x_str};{nodes_y_str};{weights_str}\n"
            
            # Write to file
            file.write(formatted_data)

    print(f"Predictions saved to {file_path}")

# Save predictions for 1st order data to a text file
save_predictions_to_text_file(y_pred_1st, merged_data_1st_order, "predictions_p1_output.txt")

# Save predictions for 2nd order data to a text file
save_predictions_to_text_file(y_pred_2nd, merged_data_2nd_order, "predictions_p2_output.txt")
