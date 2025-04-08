import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''class NodalPreprocessor(nn.Module):
    def __init__(self, num_nodes=25, domain=(-1, 1)):
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
        # Ensure batch dimension exists.
        if exp_x.dim() == 1:
            exp_x = exp_x.unsqueeze(0)
            exp_y = exp_y.unsqueeze(0)
            coeff = coeff.unsqueeze(0)
        
        # Expand nodal coordinates for broadcasting.
        X = self.X.unsqueeze(0).unsqueeze(2)  # (1, num_nodes, 1)
        Y = self.Y.unsqueeze(0).unsqueeze(2)  # (1, num_nodes, 1)
        exp_x = exp_x.unsqueeze(1)            # (batch, 1, m)
        exp_y = exp_y.unsqueeze(1)            # (batch, 1, m)
        coeff = coeff.unsqueeze(1)            # (batch, 1, m)
        
        x_terms = X ** exp_x
        y_terms = Y ** exp_y
        nodal_values = torch.sum(coeff * x_terms * y_terms, dim=2)
        
        # Normalize each sample's nodal values by dividing by its maximum value.
        max_val = nodal_values.max(dim=1, keepdim=True)[0] + 1e-6
        #nodal_values = nodal_values / max_val
        return nodal_values'''

class PreActResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(PreActResidualBlock, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.linear1 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(dim)
        self.linear2 = nn.Linear(dim, dim)
        
    def forward(self, x):
        out = self.relu(x)
        out = self.linear1(out)
        out = self.dropout(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.linear2(out)
        return x + out  # Residual connection

class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, max_output_len, num_residual=6, dropout=0.1):
        """
        Args:
            input_dim: Dimension of concatenated input [exp_x, exp_y, coeff].
            hidden_dim: Size of hidden layers.
            output_dim: Intermediate output dimension for the branches.
            max_output_len: Length of the predicted node vectors.
            num_residual: Number of pre-activation residual blocks.
            dropout: Dropout rate.
        """
        super(FeedForwardNN, self).__init__()
        # Initial input layer with BN and ReLU.
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Stack several pre-activation residual blocks.
        self.residual_blocks = nn.Sequential(
            *[PreActResidualBlock(hidden_dim, dropout) for _ in range(num_residual)]
        )
        
        # Shared layer after residual blocks.
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Branch for node predictions.
        self.nodes_branch = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True)
        )
        self.node_x_head = nn.Linear(output_dim, max_output_len)
        self.node_y_head = nn.Linear(output_dim, max_output_len)
        
        # Branch for weight predictions.
        self.weights_branch = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True)
        )
        self.weight_head = nn.Linear(output_dim, max_output_len)
    
    def forward(self, exp_x, exp_y, coeff):
        # Concatenate features.
        x = torch.cat((exp_x, exp_y, coeff), dim=1)
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        x = self.shared_layer(x)
        
        # Nodes branch: apply tanh to bound outputs in [-1,1]
        nodes_out = self.nodes_branch(x)
        pred_nodes_x = torch.tanh(self.node_x_head(nodes_out))
        pred_nodes_y = torch.tanh(self.node_y_head(nodes_out))
        
        # Weights branch: apply softplus for non-negative outputs
        weights_out = self.weights_branch(x)
        pred_weights = torch.nn.functional.softplus(self.weight_head(weights_out))
        
        return pred_nodes_x, pred_nodes_y, pred_weights

def load_ff_pipelines_model(weights_path=None):
    input_dim = 12           # [exp_x, exp_y, coeff]
    hidden_dim = 1024        # Hidden layer width
    output_dim = 1024        # Branch output dimension
    max_output_len = 16      # Maximum number of predicted nodes
    num_residual = 6         # Using 6 residual blocks
    dropout = 0.1            # Dropout rate
    model = FeedForwardNN(input_dim, hidden_dim, output_dim, max_output_len, num_residual, dropout)
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    return model

def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: {filename}, Epoch: {epoch}, Loss: {loss:.4f}")
    return epoch, loss