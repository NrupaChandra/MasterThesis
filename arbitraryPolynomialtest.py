import numpy as np
from scipy.integrate import quad
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

# Generate training data
N = 100000  # Number of training samples
max_degree = 4  # Maximum degree of the polynomial
coeffs = np.random.rand(N, max_degree + 1)  # Random coefficients
L = np.random.rand(N)  # Random lengths between 0 and 1
exact_integrals = np.zeros(N)

# Calculate exact integrals for each polynomial
for i in range(N):
    poly_coeff = coeffs[i, :]
    poly_func = lambda x: np.polyval(poly_coeff, x)
    exact_integrals[i], _ = quad(poly_func, 0, L[i])

# Prepare feature matrix
features = np.hstack((L.reshape(-1, 1), coeffs))

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, exact_integrals, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# Define the neural network
class PolynomialIntegrationNet(nn.Module):
    def __init__(self):
        super(PolynomialIntegrationNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(max_degree + 2, 10),  # Input size: max_degree + 2 (L + coefficients)
            nn.ReLU(),
            *[layer for _ in range(9) for layer in (nn.Linear(10, 10), nn.ReLU())],
            nn.Linear(10, 1)  # Output size: 1 (integral value)
        )
    
    def forward(self, x):
        return self.layers(x)

# Initialize the model, loss function, and optimizer
model = PolynomialIntegrationNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the neural network
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Validate the model
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_val_tensor)
    mse_val = criterion(y_pred_tensor, y_val_tensor).item()

print(f"Validation MSE: {mse_val}")
