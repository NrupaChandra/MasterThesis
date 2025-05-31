import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class NodalPreprocessor(nn.Module):
    def __init__(self,
                 node_x: torch.Tensor,
                 node_y: torch.Tensor):
        super().__init__()
        x = node_x.to(torch.float32)
        y = node_y.to(torch.float32)
        if x.numel() != y.numel():
            raise ValueError("node_x and node_y must have the same number of elements")
        N = x.numel()
        s = int(math.sqrt(N))
        if s * s != N:
            raise ValueError(f"Expected a square grid; got {N} points")
        self.register_buffer("X", x)
        self.register_buffer("Y", y)
        self.num_nodes = N
        self.grid_size = s

    def forward(self, exp_x, exp_y, coeff):
        if exp_x.dim() == 1:
            exp_x = exp_x.unsqueeze(0)
            exp_y = exp_y.unsqueeze(0)
            coeff = coeff.unsqueeze(0)

        X = self.X.unsqueeze(0).unsqueeze(2)  # (1, num_nodes, 1)
        Y = self.Y.unsqueeze(0).unsqueeze(2)
        exp_x = exp_x.unsqueeze(1)
        exp_y = exp_y.unsqueeze(1)
        coeff = coeff.unsqueeze(1)

        x_terms = X ** exp_x
        y_terms = Y ** exp_y
        nodal_values = torch.sum(coeff * x_terms * y_terms, dim=2)

        max_val = nodal_values.max(dim=1, keepdim=True)[0] + 1e-6
        return nodal_values / max_val


class CNN(nn.Module):

    def __init__(self,
                 dropout_rate: float = 0.0,
                 node_x: torch.Tensor = None,
                 node_y: torch.Tensor = None):
        super(CNN, self).__init__()
        self.nodal_preprocessor = NodalPreprocessor(
            node_x=node_x,
            node_y=node_y)
        self.grid_size = self.nodal_preprocessor.grid_size
        
        # Define a pure convolutional network.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()  # ensures output weights are non-negative
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, exp_x, exp_y, coeff):
        # Compute the nodal representation using the preprocessor.
        # Output shape: (batch, num_nodes)
        nodal_values = self.nodal_preprocessor(exp_x, exp_y, coeff)
        # Reshape to image format: (batch, 1, grid_size, grid_size)
        nodal_image = nodal_values.view(-1, 1, self.grid_size, self.grid_size)
        
        # Process through convolutional layers.
        x = self.relu(self.conv1(nodal_image))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        # Final convolution produces the weight grid.
        weight_grid = self.softplus(self.conv3(x))
        return weight_grid

node_x_str = """-0.9602898564975362,-0.9602898564975362,-0.9602898564975362,-0.9602898564975362,-0.9602898564975362,-0.9602898564975362,-0.9602898564975362,-0.9602898564975362,
-0.7966664774136267,-0.7966664774136267,-0.7966664774136267,-0.7966664774136267,-0.7966664774136267,-0.7966664774136267,-0.7966664774136267,-0.7966664774136267,
-0.5255324099163290,-0.5255324099163290,-0.5255324099163290,-0.5255324099163290,-0.5255324099163290,-0.5255324099163290,-0.5255324099163290,-0.5255324099163290,
-0.1834346424956498,-0.1834346424956498,-0.1834346424956498,-0.1834346424956498,-0.1834346424956498,-0.1834346424956498,-0.1834346424956498,-0.1834346424956498,
0.1834346424956499,0.1834346424956499,0.1834346424956499,0.1834346424956499,0.1834346424956499,0.1834346424956499,0.1834346424956499,0.1834346424956499,
0.5255324099163290,0.5255324099163290,0.5255324099163290,0.5255324099163290,0.5255324099163290,0.5255324099163290,0.5255324099163290,0.5255324099163290,
0.7966664774136267,0.7966664774136267,0.7966664774136267,0.7966664774136267,0.7966664774136267,0.7966664774136267,0.7966664774136267,0.7966664774136267,
0.9602898564975362,0.9602898564975362,0.9602898564975362,0.9602898564975362,0.9602898564975362,0.9602898564975362,0.9602898564975362,0.9602898564975362"""

node_y_str = """-0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956499,0.5255324099163290,0.7966664774136267,0.9602898564975362,
-0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956499,0.5255324099163290,0.7966664774136267,0.9602898564975362,
-0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956499,0.5255324099163290,0.7966664774136267,0.9602898564975362,
-0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956499,0.5255324099163290,0.7966664774136267,0.9602898564975362,
-0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956499,0.5255324099163290,0.7966664774136267,0.9602898564975362,
-0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956499,0.5255324099163290,0.7966664774136267,0.9602898564975362,
-0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956499,0.5255324099163290,0.7966664774136267,0.9602898564975362,
-0.9602898564975362,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,0.1834346424956499,0.5255324099163290,0.7966664774136267,0.9602898564975362"""

node_x_list = [float(v) for v in node_x_str.replace("\n", "").split(",") if v]
node_y_list = [float(v) for v in node_y_str.replace("\n", "").split(",") if v]

tx = torch.tensor(node_x_list, dtype=torch.float32)
ty = torch.tensor(node_y_list, dtype=torch.float32)

def load_shallow_cnn_model(weights_path=None,
                           node_x=tx,
                           node_y=ty,
                           dropout_rate=0.0):
    model = CNN(
                node_x=tx,
                node_y=ty,
                dropout_rate=dropout_rate)
    model = model.float()
    if weights_path:
        # force all tensors onto CPU
        state = torch.load(weights_path, map_location=torch.device('cpu'))
        model.load_state_dict(state)
    return model


# --- Checkpoint Saving/Loading Functions ---

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


