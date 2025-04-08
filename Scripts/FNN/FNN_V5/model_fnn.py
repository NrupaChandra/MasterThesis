import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NodalPreprocessor(nn.Module):
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
        nodal_values = nodal_values / max_val
        return nodal_values

class FeedForwardNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_output_len, num_nodes=25, domain=(-1,1),
                 dropout_rate=0.0, num_shared_layers=1):
        super(FeedForwardNN, self).__init__()
        self.nodal_preprocessor = NodalPreprocessor(num_nodes=num_nodes, domain=domain)
        input_dim = num_nodes

        # Build shared layers dynamically based on num_shared_layers.
        shared_layers = []
        in_dim = input_dim
        for _ in range(num_shared_layers):
            shared_layers.append(nn.Linear(in_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            if dropout_rate > 0:
                shared_layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        self.shared_layer = nn.Sequential(*shared_layers)
        
        # Branch for predicting node positions in x.
        self.x_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_output_len),
            nn.Tanh()
        )
        
        # Branch for predicting node positions in y.
        self.y_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_output_len),
            nn.Tanh()
        )
        
        # Branch for predicting weights.
        self.w_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, max_output_len),
            nn.Softplus()
        )
  
    def forward(self, exp_x, exp_y, coeff):
        # Obtain the nodal representation.
        x = self.nodal_preprocessor(exp_x, exp_y, coeff)
        
        # Pass through the shared layers.
        shared_features = self.shared_layer(x)
        
        # Process through individual branches.
        pred_nodes_x = self.x_layer(shared_features)
        pred_nodes_y = self.y_layer(shared_features)
        pred_weights = self.w_layer(shared_features)
        
        return pred_nodes_x, pred_nodes_y, pred_weights

def load_ff_pipelines_model(weights_path=None, hidden_dim=1024, output_dim=1024, max_output_len=16,
                             num_nodes=25, domain=(-1,1), dropout_rate=0.0, num_shared_layers=1):
    model = FeedForwardNN(hidden_dim, output_dim, max_output_len, num_nodes, domain, dropout_rate, num_shared_layers)
    model = model.float()
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
