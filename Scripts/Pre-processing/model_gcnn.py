import torch
import torch.nn.functional as F
from torch import nn
# Requires PyTorch Geometric
from torch_geometric.nn import GCNConv
import numpy as np

class NodalPreprocessor(nn.Module):
    def __init__(self, node_x, node_y):
        super().__init__()
        # If positions are provided as strings, parse them into tensors
        if isinstance(node_x, str):
            # remove newlines and split
            parts = [p.strip() for p in node_x.replace('\n', '').split(',') if p.strip()]
            node_x = torch.tensor([float(p) for p in parts], dtype=torch.float32)
        if isinstance(node_y, str):
            parts = [p.strip() for p in node_y.replace('\n', '').split(',') if p.strip()]
            node_y = torch.tensor([float(p) for p in parts], dtype=torch.float32)

        # Validate inputs are now 1D tensors
        assert isinstance(node_x, torch.Tensor) and node_x.dim() == 1, \
            "node_x must be a 1D tensor or comma-separated string"
        assert isinstance(node_y, torch.Tensor) and node_y.dim() == 1, \
            "node_y must be a 1D tensor or comma-separated string"
        assert node_x.numel() == node_y.numel(), \
            "node_x and node_y must have the same number of elements"

        self.num_nodes = node_x.numel()
        # Register buffers so they move with the model
        self.register_buffer("X", node_x)
        self.register_buffer("Y", node_y)

    def forward(self, exp_x, exp_y, coeff):
        # Ensure batch dimension exists.
        if exp_x.dim() == 1:
            exp_x = exp_x.unsqueeze(0)
            exp_y = exp_y.unsqueeze(0)
            coeff = coeff.unsqueeze(0)

        # Prepare for broadcasting
        X = self.X.unsqueeze(0).unsqueeze(2)  # (1, P, 1)
        Y = self.Y.unsqueeze(0).unsqueeze(2)  # (1, P, 1)
        exp_x = exp_x.unsqueeze(1)            # (B, 1, m)
        exp_y = exp_y.unsqueeze(1)
        coeff = coeff.unsqueeze(1)

        # Compute nodal values
        x_terms = X ** exp_x
        y_terms = Y ** exp_y
        nodal_values = torch.sum(coeff * x_terms * y_terms, dim=2)

        # Normalize per-sample
        max_val = nodal_values.max(dim=1, keepdim=True)[0] + 1e-6
        return nodal_values / max_val



class GraphQuadratureNet(nn.Module):
  
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        # Message-passing layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Dropout for feature regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Head for predicting positional shifts (dx, dy)
        self.shift_head = nn.Linear(hidden_channels, 2)

        # Head for predicting positive weights
        self.weight_head = nn.Sequential(
            nn.Linear(hidden_channels, 1),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor):
        # x: [total_nodes, in_channels]
        # edge_index: [2, E]
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        # Predict shifts and weights per node
        shifts = self.shift_head(x)               # [total_nodes, 2]
        weights = self.weight_head(x).squeeze(-1) # [total_nodes]
        return shifts, weights


def load_gnn_model(
    in_channels: int = 3,
    hidden_channels: int = 64,
    num_layers: int = 3,
    dropout_rate: float = 0.0,
    device: torch.device = None
) -> GraphQuadratureNet:
    """
    Utility to create and optionally move the GNN model to a device.

    Args:
      in_channels: number of features per node (e.g., [x,y] + any global broadcast)
      hidden_channels: hidden dimension in each message-passing layer
      num_layers: number of GCNConv layers
      dropout_rate: dropout probability after each layer
      device: torch.device to move model onto

    Returns:
      Initialized GraphQuadratureNet
    """
    model = GraphQuadratureNet(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    )
    if device is not None:
        model = model.to(device)
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


def load_checkpoint(
    model: nn.Module,
    optimizer,
    filename="checkpoint.pth"):

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: {filename}, Epoch: {epoch}, Loss: {loss:.4f}")
    return epoch, loss

