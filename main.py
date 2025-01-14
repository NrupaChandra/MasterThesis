import numpy as np
from reader import merged_data_1st_order
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Function to prepare the feature matrix
def prepare_feature_matrix(data):
    feature_matrix = []
    for entry in data:
        print(len(entry['exp_x']))
        feature_vector = entry['exp_x'] + entry['exp_y'] + entry['coeff']
        print(len(feature_vector))
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

# Define the CNN model with a Masking layer
def create_cnn_model(input_shape, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Masking(mask_value=0, input_shape=input_shape),  # Masking the padded values
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(output_dim)  # Output layer
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model

# Prepare feature and target matrices for nodes_x, nodes_y, and weights
feature_matrix = prepare_feature_matrix(merged_data_1st_order)
exit()
target_matrix_x, max_length_x = prepare_target_matrix(merged_data_1st_order, 'nodes_x')
target_matrix_y, max_length_y = prepare_target_matrix(merged_data_1st_order, 'nodes_y')
target_matrix_weights, max_length_weights = prepare_target_matrix(merged_data_1st_order, 'weights')

# Split data into training and testing sets
X_train, X_test, y_train_x, y_test_x = train_test_split(feature_matrix, target_matrix_x, test_size=0.2, random_state=42)
_, _, y_train_y, y_test_y = train_test_split(feature_matrix, target_matrix_y, test_size=0.2, random_state=42)
_, _, y_train_weights, y_test_weights = train_test_split(feature_matrix, target_matrix_weights, test_size=0.2, random_state=42)

# Reshape feature matrix for CNN input
X_train_cnn = X_train[..., np.newaxis]  # Add a channel dimension
X_test_cnn = X_test[..., np.newaxis]

# Train CNN for nodes_x
model_x = create_cnn_model((X_train_cnn.shape[1], 1), max_length_x)
model_x.fit(X_train_cnn, y_train_x, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
y_pred_x = model_x.predict(X_test_cnn)
mse_x = mean_squared_error(y_test_x, y_pred_x)
print("Mean Squared Error for Nodes X:", mse_x)

# Train CNN for nodes_y
model_y = create_cnn_model((X_train_cnn.shape[1], 1), max_length_y)
model_y.fit(X_train_cnn, y_train_y, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
y_pred_y = model_y.predict(X_test_cnn)
mse_y = mean_squared_error(y_test_y, y_pred_y)
print("Mean Squared Error for Nodes Y:", mse_y)

# Train CNN for weights
model_weights = create_cnn_model((X_train_cnn.shape[1], 1), max_length_weights)
model_weights.fit(X_train_cnn, y_train_weights, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
y_pred_weights = model_weights.predict(X_test_cnn)
mse_weights = mean_squared_error(y_test_weights, y_pred_weights)
print("Mean Squared Error for Weights:", mse_weights)

# Function to save predictions to a text file in the specified format
def save_predictions_to_text_file(pred_x, pred_y, pred_weights, test_data, file_path, max_len_x, max_len_y, max_len_weights):
    with open(file_path, 'w') as file:
        # Write the header
        file.write("number;id;nodes_x;nodes_y;weights\n")

        # Write the predictions and corresponding IDs
        for idx, (entry, px, py, pw) in enumerate(zip(test_data, pred_x, pred_y, pred_weights)):
            entry_id = entry['id']

            # Extract original lengths
            num_nodes_x = len(entry['nodes_x'])
            num_nodes_y = len(entry['nodes_y'])
            num_weights = len(entry['weights'])

            # Extract predicted values up to the original length
            predicted_nodes_x = px[:num_nodes_x].tolist()
            predicted_nodes_y = py[:num_nodes_y].tolist()
            predicted_weights = pw[:num_weights].tolist()

            # Format the data
            nodes_x_str = ",".join(map(str, predicted_nodes_x))
            nodes_y_str = ",".join(map(str, predicted_nodes_y))
            weights_str = ",".join(map(str, predicted_weights))

            # Write the formatted line
            file.write(f"{idx};{entry_id};{nodes_x_str};{nodes_y_str};{weights_str}\n")

    print(f"Predictions saved to {file_path}")

# Save predictions to a text file
save_predictions_to_text_file(y_pred_x, y_pred_y, y_pred_weights, merged_data_1st_order,
                              "cnn_predictions_p1_output.txt", max_length_x, max_length_y, max_length_weights)
