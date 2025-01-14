import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from reader import merged_data_1st_order

# Function to prepare the feature matrix
def prepare_feature_matrix(data):
    feature_matrix = []
    for entry in data:
        feature_vector = entry['exp_x'] + entry['exp_y'] + entry['coeff']
        feature_matrix.append(feature_vector)
    return np.array(feature_matrix)

# Function to prepare the target matrix with padding
def prepare_target_matrix(data, target_key):
    target_matrix = []
    max_length = max(len(entry[target_key]) for entry in data)
    for entry in data:
        padded_target = entry[target_key] + [0] * (max_length - len(entry[target_key]))
        target_matrix.append(padded_target)
    return np.array(target_matrix), max_length

# Define the DNN model with separate outputs for nodes and weights
def create_dnn_model(input_dim, output_dim_nodes, output_dim_weights):
    input_layer = tf.keras.Input(shape=(input_dim,))
    
    # Shared hidden layers
    shared = tf.keras.layers.Dense(128, activation='relu')(input_layer)
    shared = tf.keras.layers.Dense(128, activation='relu')(shared)
    
    # Separate output layers for nodes and weights
    output_nodes = tf.keras.layers.Dense(output_dim_nodes, name='nodes_output')(shared)
    output_weights = tf.keras.layers.Dense(output_dim_weights, name='weights_output')(shared)
    
    # Model definition
    model = tf.keras.Model(inputs=input_layer, outputs=[output_nodes, output_weights])
    model.compile(optimizer='adam', 
                  loss={'nodes_output': 'mse', 'weights_output': 'mse'}, 
                  metrics={'nodes_output': 'mse', 'weights_output': 'mse'})
    return model

# Load and preprocess the dataset
feature_matrix = prepare_feature_matrix(merged_data_1st_order)
target_matrix_nodes, max_length_nodes = prepare_target_matrix(merged_data_1st_order, 'nodes_x')
target_matrix_weights, max_length_weights = prepare_target_matrix(merged_data_1st_order, 'weights')

# Split data into training and testing sets
X_train, X_test, y_train_nodes, y_test_nodes = train_test_split(feature_matrix, target_matrix_nodes, test_size=0.2, random_state=42)
_, _, y_train_weights, y_test_weights = train_test_split(feature_matrix, target_matrix_weights, test_size=0.2, random_state=42)

# Train the DNN model
model = create_dnn_model(X_train.shape[1], max_length_nodes, max_length_weights)
history = model.fit(
    X_train, 
    {'nodes_output': y_train_nodes, 'weights_output': y_train_weights},
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=1
)

# Predict and evaluate
predictions = model.predict(X_test)
pred_nodes, pred_weights = predictions[0], predictions[1]
mse_nodes = mean_squared_error(y_test_nodes, pred_nodes)
mse_weights = mean_squared_error(y_test_weights, pred_weights)
print("Mean Squared Error for Nodes:", mse_nodes)
print("Mean Squared Error for Weights:", mse_weights)

# Save predictions to a text file
def save_predictions_to_text_file(pred_nodes, pred_weights, test_data, file_path):
    with open(file_path, 'w') as file:
        file.write("number;id;nodes_x;nodes_y;weights\n")
        for idx, (entry, pn, pw) in enumerate(zip(test_data, pred_nodes, pred_weights)):
            entry_id = entry['id']
            num_nodes = len(entry['nodes_x'])  # Assuming nodes_x and nodes_y have the same length
            num_weights = len(entry['weights'])

            # Split predicted nodes into nodes_x and nodes_y
            predicted_nodes_x = pn[:num_nodes].tolist()
            predicted_nodes_y = pn[num_nodes:num_nodes * 2].tolist()
            predicted_weights = pw[:num_weights].tolist()

            # Convert lists to strings
            nodes_x_str = ",".join(map(str, predicted_nodes_x))
            nodes_y_str = ",".join(map(str, predicted_nodes_y))
            weights_str = ",".join(map(str, predicted_weights))

            # Write to file
            file.write(f"{idx};{entry_id};{nodes_x_str};{nodes_y_str};{weights_str}\n")
    print(f"Predictions saved to {file_path}")

save_predictions_to_text_file(pred_nodes, pred_weights, merged_data_1st_order, "dnn_predictions_output.txt")
