import numpy as np
from reader import merged_data_1st_order
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Gaussian Quadrature Rule to calculate the integral
def gaussian_quadrature_integral(nodes, weights):
    """
    Computes the integral using the Gaussian quadrature rule.
    Integral = Σ (weights[i] * f(nodes[i]))
    Here, f(x) = 1 as a placeholder (update this based on the actual use case).
    """
    # Replace f(x) with the appropriate function if needed
    integral = sum(w * 1 for w in weights)  # Assuming f(x) = 1 for simplicity
    return integral

# Prepare the feature matrix
def prepare_feature_matrix(data):
    feature_matrix = []
    for entry in data:
        feature_vector = entry['exp_x'] + entry['exp_y'] + entry['coeff']
        feature_matrix.append(feature_vector)
    return np.array(feature_matrix)

# Prepare the target matrix for exactly 10 nodes
def prepare_target_matrix_limited(data, target_key, limit=10):
    target_matrix = []
    for entry in data:
        limited_target = entry[target_key][:limit] + [0] * (limit - len(entry[target_key][:limit]))
        target_matrix.append(limited_target)
    return np.array(target_matrix)

# Custom loss function: compare actual and predicted integrals
def integral_loss(y_true_nodes, y_true_weights, y_pred_nodes, y_pred_weights):
    loss = 0.0
    for true_nodes, true_weights, pred_nodes, pred_weights in zip(y_true_nodes, y_true_weights, y_pred_nodes, y_pred_weights):
        actual_integral = gaussian_quadrature_integral(true_nodes, true_weights)
        predicted_integral = gaussian_quadrature_integral(pred_nodes, pred_weights)
        loss += (actual_integral - predicted_integral) ** 2
    return loss / len(y_true_nodes)

# Prepare feature and target matrices
feature_matrix = prepare_feature_matrix(merged_data_1st_order)
target_matrix_x = prepare_target_matrix_limited(merged_data_1st_order, 'nodes_x', limit=10)
target_matrix_y = prepare_target_matrix_limited(merged_data_1st_order, 'nodes_y', limit=10)
target_matrix_weights = prepare_target_matrix_limited(merged_data_1st_order, 'weights', limit=10)

# Split data into training and testing sets
X_train, X_test, y_train_x, y_test_x = train_test_split(feature_matrix, target_matrix_x, test_size=0.2, random_state=42)
_, _, y_train_y, y_test_y = train_test_split(feature_matrix, target_matrix_y, test_size=0.2, random_state=42)
_, _, y_train_weights, y_test_weights = train_test_split(feature_matrix, target_matrix_weights, test_size=0.2, random_state=42)

# Define a neural network model
def build_model(input_dim, output_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(64, activation='relu'),
        Dense(output_dim)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')  # Use MSE temporarily
    return model

# Train models for nodes_x, nodes_y, and weights
model_x = build_model(X_train.shape[1], y_train_x.shape[1])
model_x.fit(X_train, y_train_x, epochs=50, batch_size=32, validation_split=0.2)

model_y = build_model(X_train.shape[1], y_train_y.shape[1])
model_y.fit(X_train, y_train_y, epochs=50, batch_size=32, validation_split=0.2)

model_weights = build_model(X_train.shape[1], y_train_weights.shape[1])
model_weights.fit(X_train, y_train_weights, epochs=50, batch_size=32, validation_split=0.2)

# Predict
y_pred_x = model_x.predict(X_test)
y_pred_y = model_y.predict(X_test)
y_pred_weights = model_weights.predict(X_test)

# Compute the training loss using integral comparison
training_loss = integral_loss(y_test_x, y_test_weights, y_pred_x, y_pred_weights)
print("Training Loss (Integral Comparison):", training_loss)

# Save predictions
def save_predictions_to_text_file(pred_x, pred_y, pred_weights, test_data, file_path):
    output_path = r"C:\Git\MasterThesis\outputs\predictions_integral_based.txt"
    with open(output_path, 'w') as file:
        file.write("number;id;nodes_x;nodes_y;weights\n")
        for idx, (entry, px, py, pw) in enumerate(zip(test_data, pred_x, pred_y, pred_weights)):
            entry_id = entry['id']
            nodes_x_str = ",".join(map(str, px))
            nodes_y_str = ",".join(map(str, py))
            weights_str = ",".join(map(str, pw))
            file.write(f"{idx};{entry_id};{nodes_x_str};{nodes_y_str};{weights_str}\n")
    print(f"Predictions saved to {output_path}")

# Save corrected predictions
save_predictions_to_text_file(y_pred_x, y_pred_y, y_pred_weights, merged_data_1st_order,
                              "predictions_integral_based.txt")
