import numpy as np
from reader import merged_data_1st_order
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

# New function for additional feature engineering
def add_spread_features(feature_matrix):
    spread_features = np.std(feature_matrix, axis=1, keepdims=True)
    return np.hstack([feature_matrix, spread_features])

# Custom loss function to penalize clustering
def spread_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    variance_penalty = 1.0 / (tf.math.reduce_variance(y_pred) + 1e-5)
    return mse + 0.1 * variance_penalty

# Modified model creation function
def create_improved_model(input_shape, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Masking(mask_value=0, input_shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(output_dim)
    ])
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss=spread_loss, metrics=['mse'])
    return model

# Function to save predictions to a text file in the specified format
def save_predictions_to_text_file(pred_x, pred_y, pred_weights, data, file_path):
    with open(file_path, 'w') as file:
        file.write("number;id;nodes_x;nodes_y;weights\n")
        for idx, (entry, px, py, pw) in enumerate(zip(data, pred_x, pred_y, pred_weights)):
            entry_id = entry['id']
            num_nodes_x = len(entry['nodes_x'])
            num_nodes_y = len(entry['nodes_y'])
            num_weights = len(entry['weights'])
            
            predicted_nodes_x = px[:num_nodes_x].tolist()
            predicted_nodes_y = py[:num_nodes_y].tolist()
            predicted_weights = pw[:num_weights].tolist()
            
            nodes_x_str = ",".join(map(str, predicted_nodes_x))
            nodes_y_str = ",".join(map(str, predicted_nodes_y))
            weights_str = ",".join(map(str, predicted_weights))
            
            file.write(f"{idx};{entry_id};{nodes_x_str};{nodes_y_str};{weights_str}\n")
    print(f"Predictions saved to {file_path}")

# Post-processing to further spread out predictions
def spread_predictions(predictions, factor=1.2):
    mean = np.mean(predictions)
    return mean + (predictions - mean) * factor

# Prepare data
feature_matrix = prepare_feature_matrix(merged_data_1st_order)
target_matrix_x, max_length_x = prepare_target_matrix(merged_data_1st_order, 'nodes_x')
target_matrix_y, max_length_y = prepare_target_matrix(merged_data_1st_order, 'nodes_y')
target_matrix_weights, _ = prepare_target_matrix(merged_data_1st_order, 'weights')

# Feature engineering and normalization
feature_matrix = add_spread_features(feature_matrix)
scaler = StandardScaler()
feature_matrix_scaled = scaler.fit_transform(feature_matrix)

# Output scaling
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
target_matrix_x_scaled = scaler_x.fit_transform(target_matrix_x)
target_matrix_y_scaled = scaler_y.fit_transform(target_matrix_y)

# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores_x = []
mse_scores_y = []

for fold, (train_index, val_index) in enumerate(kf.split(feature_matrix_scaled), 1):
    print(f"Training fold {fold}")
    
    X_train, X_val = feature_matrix_scaled[train_index], feature_matrix_scaled[val_index]
    y_train_x, y_val_x = target_matrix_x_scaled[train_index], target_matrix_x_scaled[val_index]
    y_train_y, y_val_y = target_matrix_y_scaled[train_index], target_matrix_y_scaled[val_index]
    
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    
    model_x = create_improved_model((X_train.shape[1], 1), max_length_x)
    model_y = create_improved_model((X_train.shape[1], 1), max_length_y)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    
    model_x.fit(X_train, y_train_x, epochs=100, batch_size=32, validation_data=(X_val, y_val_x), callbacks=[early_stopping], verbose=1)
    model_y.fit(X_train, y_train_y, epochs=100, batch_size=32, validation_data=(X_val, y_val_y), callbacks=[early_stopping], verbose=1)
    
    y_pred_x = model_x.predict(X_val)
    y_pred_y = model_y.predict(X_val)
    
    # Inverse transform predictions
    y_pred_x = scaler_x.inverse_transform(y_pred_x)
    y_pred_y = scaler_y.inverse_transform(y_pred_y)
    y_val_x = scaler_x.inverse_transform(y_val_x)
    y_val_y = scaler_y.inverse_transform(y_val_y)
    
    mse_x = mean_squared_error(y_val_x, y_pred_x)
    mse_y = mean_squared_error(y_val_y, y_pred_y)
    
    mse_scores_x.append(mse_x)
    mse_scores_y.append(mse_y)
    
    print(f"Fold {fold} - MSE for Nodes X: {mse_x}")
    print(f"Fold {fold} - MSE for Nodes Y: {mse_y}")

print(f"Average MSE for Nodes X: {np.mean(mse_scores_x)}")
print(f"Average MSE for Nodes Y: {np.mean(mse_scores_y)}")

# Train final models on the entire dataset
X_train_final = feature_matrix_scaled[..., np.newaxis]

final_model_x = create_improved_model((X_train_final.shape[1], 1), max_length_x)
final_model_y = create_improved_model((X_train_final.shape[1], 1), max_length_y)

final_model_x.fit(X_train_final, target_matrix_x_scaled, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)
final_model_y.fit(X_train_final, target_matrix_y_scaled, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

# Make predictions on the entire dataset
X_all = feature_matrix_scaled[..., np.newaxis]
y_pred_x_all = final_model_x.predict(X_all)
y_pred_y_all = final_model_y.predict(X_all)

# Inverse transform predictions
y_pred_x_all = scaler_x.inverse_transform(y_pred_x_all)
y_pred_y_all = scaler_y.inverse_transform(y_pred_y_all)

# Apply post-processing to spread out predictions
y_pred_x_all = spread_predictions(y_pred_x_all)
y_pred_y_all = spread_predictions(y_pred_y_all)

# Save predictions
save_predictions_to_text_file(y_pred_x_all, y_pred_y_all, target_matrix_weights, merged_data_1st_order, "improved_spread_predictions_output.txt")
