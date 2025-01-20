import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_layers, max_output_len):
        
        super().__init__()

        self.embedding = nn.Linear(input_dim, embed_dim)

        # Transformer Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim),
            num_layers=num_layers
        )
        
        # Separate heads for x-nodes, y-nodes, and weights
        self.node_x_head = nn.Linear(embed_dim, max_output_len)
        self.node_y_head = nn.Linear(embed_dim, max_output_len)
        self.weight_head = nn.Linear(embed_dim, max_output_len)

    def forward(self, exp_x, exp_y, coeff, _):

        # concatenate input vectors
        x = torch.cat((exp_x, exp_y, coeff), 1)

        x = self.embedding(x).unsqueeze(1)  # Add sequence length dimension

        encoded = self.encoder(x).squeeze(1)  # Remove sequence length dimension

        # Process output for nodes and weights, ensure positivity of weights and interval [-1,1] for nodes
        pred_nodes_x = torch.tanh(self.node_x_head(encoded))  # Map to [-1, 1]
        pred_nodes_y = torch.tanh(self.node_y_head(encoded))  # Map to [-1, 1]
        pred_weights = torch.nn.functional.softplus(self.weight_head(encoded))  # Ensure positivity

        return pred_nodes_x, pred_nodes_y, pred_weights
    
def load_model(weights_path=None):
        input_dim = 12          # Size of concatenated [exp_x, exp_y, coeff] vectors
        embed_dim = 16         # Embedding size for Transformer
        num_heads = 4           # Number of attention heads
        ff_dim = 32              # Feedforward network size
        num_layers = 2          # Number of Transformer layers
        max_output_len = 16     # Maximum length of quadrature rule outputs

        model = TransformerModel(input_dim, embed_dim, num_heads, ff_dim, num_layers, max_output_len)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("Number of parameters in model:" + str(pytorch_total_params))

        if weights_path:
                # Modellgewichte laden, wenn ein Pfad angegeben ist
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