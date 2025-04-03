import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NodalPreprocessor(nn.Module):
    def __init__(self, num_nodes=1225, domain=(-1, 1)):
        super(NodalPreprocessor, self).__init__()
        self.num_nodes = num_nodes
        self.domain = domain
        self.grid_size = int(np.sqrt(num_nodes))
        if self.grid_size ** 2 != num_nodes:
            raise ValueError("num_nodes must be a perfect square (e.g., 4, 9, 16, ...)")
        
        xs = torch.linspace(domain[0], domain[1], self.grid_size, dtype=torch.float32)
        ys = torch.linspace(domain[0], domain[1], self.grid_size, dtype=torch.float32)
        X, Y = torch.meshgrid(xs, ys, indexing='ij')
        self.register_buffer("X", X.flatten())
        self.register_buffer("Y", Y.flatten())

    def forward(self, exp_x, exp_y, coeff):
        if exp_x.dim() == 1:
            exp_x = exp_x.unsqueeze(0)
            exp_y = exp_y.unsqueeze(0)
            coeff = coeff.unsqueeze(0)
        
        X = self.X.unsqueeze(0).unsqueeze(2)  # (1, num_nodes, 1)
        Y = self.Y.unsqueeze(0).unsqueeze(2)  # (1, num_nodes, 1)
        exp_x = exp_x.unsqueeze(1)            # (batch, 1, m)
        exp_y = exp_y.unsqueeze(1)            # (batch, 1, m)
        coeff = coeff.unsqueeze(1)            # (batch, 1, m)
        
        x_terms = X ** exp_x
        y_terms = Y ** exp_y
        nodal_values = torch.sum(coeff * x_terms * y_terms, dim=2)
        
        max_val = nodal_values.max(dim=1, keepdim=True)[0] + 1e-6
        nodal_values = nodal_values / max_val
        return nodal_values

class CNN_Weights_and_NodeCorrections(nn.Module):
    """
    A CNN model that uses the nodal representation from a 35x35 grid (via the nodal preprocessor)
    and outputs predictions on a fixed 16x16 grid:
      - A weight for each node (non-negative).
      - A delta correction (for x and y) for each node.
    """
    def __init__(self, num_nodes, domain=(-1,1), dropout_rate=0.0):
        super(CNN_Weights_and_NodeCorrections, self).__init__()
        self.nodal_preprocessor = NodalPreprocessor(num_nodes=num_nodes, domain=domain)
        self.input_grid_size = int(np.sqrt(num_nodes))  
        self.output_grid_size = 16
        
        # Shared convolutional layers.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        
        # Two separate branches after feature extraction:
        # Weight branch:
        self.conv3_weights = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        # Delta correction branch (2 channels: one for Δx and one for Δy):
        self.conv3_deltas = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()  # for weights to be non-negative
        self.tanh = nn.Tanh()          # for delta corrections (bounded output)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, exp_x, exp_y, coeff):
        # Compute nodal representation (35x35 grid flattened to a vector).
        nodal_values = self.nodal_preprocessor(exp_x, exp_y, coeff)
        nodal_image = nodal_values.view(-1, 1, self.input_grid_size, self.input_grid_size)
        
        # Shared feature extraction.
        x = self.relu(self.conv1(nodal_image))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        
        # Adapt the features to the fixed 16x16 grid using adaptive pooling.
        x_pooled = F.adaptive_avg_pool2d(x, (self.output_grid_size, self.output_grid_size))
        
        # Branch for weight prediction.
        weight_out = self.softplus(self.conv3_weights(x_pooled))  # (batch, 1, 16, 16)
        
        # Branch for node correction prediction.
        delta_out = self.tanh(self.conv3_deltas(x_pooled))          # (batch, 2, 16, 16)
        
        return weight_out, delta_out


def load_shallow_cnn_model(weights_path=None, num_nodes=1225, domain=(-1,1), dropout_rate=0.0):
    """
    Instantiates the pure CNN model that predicts the grid of weights.
    """
    model = CNN_Weights_and_NodeCorrections(num_nodes=num_nodes, domain=domain, dropout_rate=dropout_rate)
    model = model.float()
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    return model

def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth"):
    """
    Saves a checkpoint containing the model state dict, optimizer state dict,
    epoch, and loss.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    """
    Loads a checkpoint and restores the model and optimizer state dictionaries,
    as well as returning the epoch and loss.
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: {filename}, Epoch: {epoch}, Loss: {loss:.4f}")
    return epoch, loss
