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
            raise ValueError("num_nodes must be a perfect square (e.g., 4, 9, ...)")
        
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

class CNN_FNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_output_len, num_nodes=25, domain=(-1,1), dropout_rate=0.0):
        super(CNN_FNN, self).__init__()
        self.nodal_preprocessor = NodalPreprocessor(num_nodes=num_nodes, domain=domain)
        self.grid_size = int(np.sqrt(num_nodes))  # e.g., 5 for 25 nodes
        
        # A single convolutional layer.
        self.conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1) # input channels 1 because we are treating it as a grey scale image 
        self.relu = nn.ReLU()
        
        # Flattened feature dimension is 8 * grid_size * grid_size.
        flattened_dim = 8 * self.grid_size * self.grid_size
        
        # One fully connected shared layer.
        self.fc_shared = nn.Sequential(
            nn.Linear(flattened_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        
        # Branch for predicting node positions in x.
        self.node_x_branch = nn.Sequential(
            nn.Linear(hidden_dim, max_output_len),
            nn.Tanh()  # ensures outputs are in [-1, 1]
        )
        
        # Branch for predicting node positions in y.
        self.node_y_branch = nn.Sequential(
            nn.Linear(hidden_dim, max_output_len),
            nn.Tanh()
        )
        
        # Branch for predicting weights.
        self.weight_branch = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, max_output_len),
            nn.Softplus()  # ensures weights are positive
        )
    
    def forward(self, exp_x, exp_y, coeff):
        # Get the nodal representation from the preprocessor.
        nodal_values = self.nodal_preprocessor(exp_x, exp_y, coeff)
        # Reshape to (batch, 1, grid_size, grid_size) standart input format for CNN in pytorch (batch_size, channels, hight, width) (-1 tells pytorch to automatically infer this dimension based on the toal number of elements in the tensor)
        nodal_image = nodal_values.view(-1, 1, self.grid_size, self.grid_size)
        
        # Apply the convolutional layer.
        conv_out = self.relu(self.conv(nodal_image))
        
        # Flatten the features.
        flat_features = conv_out.view(conv_out.size(0), -1)
        
        # Pass through the fully connected shared layer.
        shared_features = self.fc_shared(flat_features)
        
        # Branch into predictions.
        pred_nodes_x = self.node_x_branch(shared_features)
        pred_nodes_y = self.node_y_branch(shared_features)
        pred_weights = self.weight_branch(shared_features)
        
        return pred_nodes_x, pred_nodes_y, pred_weights

def load_shallow_cnn_model(weights_path=None, hidden_dim=256, output_dim=256, max_output_len=256,
                           num_nodes=25, domain=(-1,1), dropout_rate=0.0):
    model = CNN_FNN(hidden_dim, output_dim, max_output_len,
                    num_nodes=num_nodes, domain=domain, dropout_rate=dropout_rate)
    model = model.float()
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    return model

# Create the model instance (for example usage)
model = load_shallow_cnn_model(hidden_dim=256, output_dim=256, max_output_len=256,
                               num_nodes=25, domain=(-1,1), dropout_rate=0.0)

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


