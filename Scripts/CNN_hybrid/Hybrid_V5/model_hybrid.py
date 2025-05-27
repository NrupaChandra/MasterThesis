
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NodalPreprocessor(nn.Module):
    def __init__(self,
                 num_nodes=64,
                 domain=(-1, 1),
                 node_x_str: str = None,
                 node_y_str: str = None):
        super(NodalPreprocessor, self).__init__()
        self.num_nodes = num_nodes
        self.domain = domain
        self.grid_size = int(np.sqrt(num_nodes))
     # Parse custom node positions
        x_vals = list(map(float, node_x_str.strip().split(',')))
        y_vals = list(map(float, node_y_str.strip().split(',')))
            
        X = torch.tensor(x_vals, dtype=torch.float32)
        Y = torch.tensor(y_vals, dtype=torch.float32)
    
        # Register as buffers for lightning-fast access and serialization
        self.register_buffer("X", X)
        self.register_buffer("Y", Y)

    def forward(self, exp_x, exp_y, coeff):
        if exp_x.dim() == 1:
            exp_x = exp_x.unsqueeze(0)
            exp_y = exp_y.unsqueeze(0)
            coeff = coeff.unsqueeze(0)

        # (batch, num_nodes, m_terms)
        X = self.X.unsqueeze(0).unsqueeze(2)
        Y = self.Y.unsqueeze(0).unsqueeze(2)
        exp_x = exp_x.unsqueeze(1)
        exp_y = exp_y.unsqueeze(1)
        coeff = coeff.unsqueeze(1)

        x_terms = X ** exp_x
        y_terms = Y ** exp_y
        nodal_values = torch.sum(coeff * x_terms * y_terms, dim=2)

        max_val = nodal_values.max(dim=1, keepdim=True)[0] + 1e-6
        return nodal_values / max_val

class CNN_FNN(nn.Module):
    def __init__(self,
                 hidden_dim,
                 output_dim,
                 max_output_len,
                 num_nodes=64,
                 domain=(-1,1),
                 dropout_rate=0.0,
                 node_x_str: str = None,
                 node_y_str: str = None):
        super(CNN_FNN, self).__init__()
        # Pass custom nodes into preprocessor
        self.nodal_preprocessor = NodalPreprocessor(
            num_nodes=num_nodes,
            domain=domain,
            node_x_str=node_x_str,
            node_y_str=node_y_str
        )
        self.grid_size = int(np.sqrt(num_nodes))

        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        flattened_dim = 8 * self.grid_size * self.grid_size

        self.fc_shared = nn.Sequential(
            nn.Linear(flattened_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )

        self.node_x_branch = nn.Sequential(
            nn.Linear(hidden_dim, max_output_len),
            nn.Tanh()
        )
        self.node_y_branch = nn.Sequential(
            nn.Linear(hidden_dim, max_output_len),
            nn.Tanh()
        )
        self.weight_branch = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, max_output_len),
            nn.Softplus()
        )

    def forward(self, exp_x, exp_y, coeff):
        nodal_values = self.nodal_preprocessor(exp_x, exp_y, coeff)
        img = nodal_values.view(-1, 1, self.grid_size, self.grid_size)
        conv_out = self.relu(self.conv(img))
        flat = conv_out.view(conv_out.size(0), -1)
        shared = self.fc_shared(flat)
        return (
            self.node_x_branch(shared),
            self.node_y_branch(shared),
            self.weight_branch(shared)
        )

# Example instantiation with your custom 8x8 nodes:
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

model = CNN_FNN(
    hidden_dim=256,
    output_dim=256,
    max_output_len=64,
    num_nodes=64,
    domain=(-1,1),
    dropout_rate=0.0,
    node_x_str=node_x_str,
    node_y_str=node_y_str
)

def load_shallow_cnn_model(
    weights_path=None,
    hidden_dim=256,
    output_dim=256,
    max_output_len=64,
    num_nodes=64,
    domain=(-1,1),
    dropout_rate=0.0,
    node_x_str: str = None,
    node_y_str: str = None
):
    model = CNN_FNN(
        hidden_dim,
        output_dim,
        max_output_len,
        num_nodes=num_nodes,
        domain=domain,
        dropout_rate=dropout_rate,
        node_x_str=node_x_str,
        node_y_str=node_y_str
    )
    model = model.float()
    if weights_path:
        # model.load_state_dict(torch.load(weights_path))
        # force all CUDA tensors onto CPU
        state = torch.load(weights_path, map_location=torch.device('cpu'))
        model.load_state_dict(state)
    return model


# Create the model instance (for example usage)
model = load_shallow_cnn_model(
    weights_path=None,
    hidden_dim=256,
    output_dim=256,
    max_output_len=64,
    num_nodes=64,
    domain=(-1,1),
    dropout_rate=0.0,
    node_x_str=node_x_str,
    node_y_str=node_y_str
)

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


