#!/usr/bin/env python
import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from multidataloader_fnn import MultiChunkDataset  # Your dataset class
from model_gcnn import load_gnn_model, NodalPreprocessor, save_checkpoint, load_checkpoint
from torch_geometric.nn import knn_graph
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt

# Set default dtype to single precision
torch.set_default_dtype(torch.float32)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Paths
data_dir  = "/work/scratch/ng66sume/Root/Data/"
model_dir = "/work/scratch/ng66sume/Models/GCNN/GCNN_v3/"
os.makedirs(model_dir, exist_ok=True)

# Hyperparameters
in_channels   = 3           # [x, y, nodal_value]
hidden_channels = 64
num_layers    = 5
dropout_rate  = 0.0014281973712297197
lr            = 0.00019023467549488614
weight_decay  = 0.0027813681941344565
epochs        = 500
batch_size    = 1          # one cell per batch
k_neighbors   = 4          # for kNN graph connectivity
w_level_set = 0.8721404439168534

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

# Utility: test functions (Legendre basis)
def legendre_poly(n, x):
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return x
    else:
        P_nm2 = torch.ones_like(x)
        P_nm1 = x
        for k in range(1, n):
            P_n = ((2 * k + 1) * x * P_nm1 - k * P_nm2) / (k + 1)
            P_nm2, P_nm1 = P_nm1, P_n
        return P_n

def test_functions():
    funcs = []
    for i in range(3):
        for j in range(3):
            funcs.append(lambda x, y, i=i, j=j: legendre_poly(i, x) * legendre_poly(j, y))
    return funcs

def evaluate_levelset(exp_x, exp_y, coeff,
                      pred_nodes_x, pred_nodes_y):
    # pred_nodes_x, pred_nodes_y: [batch, P]
    # exp_x, exp_y, coeff:         [M]

    # 1) lift nodes → [batch, P, 1]
    x = pred_nodes_x.unsqueeze(2)  
    y = pred_nodes_y.unsqueeze(2)

    # 2) lift exponents → [1, 1, M]
    ex = exp_x.view(1, 1, -1)      
    ey = exp_y.view(1, 1, -1)

    # 3) lift coefficients → [1, 1, M]
    c  = coeff.view(1,  1, -1)     

    # 4) broadcasted elementwise power & multiply → [batch, P, M]
    values = c * (x**ex * y**ey)

    # 5) sum over the M monomials → [batch, P]
    return values.sum(dim=2)


# Integration loss functions 
criterion = nn.MSELoss()

def evaluate_integral(exp_x, exp_y, coeff, pred_x, pred_y, pred_w, true_x, true_y, true_w):
    penalty = 0.0
    integration_loss = 0.0
    for test_fn in test_functions():
        pred_I = torch.sum(pred_w * test_fn(pred_x, pred_y), dim=1)
        true_I = torch.sum(true_w * test_fn(true_x, true_y), dim=1)
        integration_loss += criterion(pred_I, true_I)
    integration_loss /= len(test_functions())
    levelset_values = evaluate_levelset(exp_x, exp_y, coeff, pred_x, pred_y)
    penalty += torch.sum(torch.relu(levelset_values))
    penalty /= exp_x.size(0)
    total_loss =  integration_loss +  w_level_set*penalty
    return total_loss

# Build grid positions and edges once
domain      = (-1.0, 1.0)
preprocessor = NodalPreprocessor(node_x_str, node_y_str).to(device)
pos          = torch.stack([preprocessor.X, preprocessor.Y], dim=1).to(device)  # [P,2]
edge_index   = knn_graph(pos, k=k_neighbors, loop=False).to(device)

# Model instantiation
model     = load_gnn_model(in_channels, hidden_channels, num_layers, dropout_rate, device)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# DataLoader
dataset     = MultiChunkDataset(
    index_file=os.path.join(data_dir, 'combined_preprocessed_chunks_10kBernstein/index.txt'),
    base_dir=data_dir
)
print("Dataset size:", len(dataset))
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

def collate_fn(batch):
    exp_x, exp_y, coeff, true_x, true_y, true_w, masks = batch[0]
    return exp_x, exp_y, coeff, true_x, true_y, true_w

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Training loop
def train_gcnn(model, train_dataloader, val_dataloader, optimizer, criterion, test_functions,
               epochs, checkpoint_path=None, save_every=5):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(model_dir, "gcnn_checkpoint.pth")
    model.to(device)
    train_losses, val_losses = [], []
    epoch_list, epoch_times = [], []
    start_epoch = 0
    # Scheduler over all steps
    total_steps = epochs * len(train_dataloader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='linear',
        final_div_factor=100,
        verbose=True
    )
    # Resume if checkpoint exists
    if os.path.exists(checkpoint_path):
        print("Checkpoint found. Loading checkpoint...")
        start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        print("No checkpoint found. Starting training from epoch 1.")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        torch.cuda.empty_cache()
        epoch_list.append(epoch + 1)

        # --- Training ---
        model.train()
        train_loss = 0.0
        for batch_idx, (exp_x, exp_y, coeff, true_x, true_y, true_w) in enumerate(train_dataloader):
            # move all tensors to device
            exp_x, exp_y, coeff, true_x, true_y, true_w = (
                x.to(device) for x in (exp_x, exp_y, coeff, true_x, true_y, true_w)
            )

            optimizer.zero_grad()

            # compute nodal values on grid
            nodal_vals = preprocessor(
                exp_x.unsqueeze(0),
                exp_y.unsqueeze(0),
                coeff.unsqueeze(0)
            ).squeeze(0)  # → [P]

            node_feat = torch.cat([pos, nodal_vals.unsqueeze(1)], dim=1)  # [P,3]
            shifts, weights = model(node_feat, edge_index)

            pred_nodes = pos + shifts        # [P,2]
            pred_x = pred_nodes[:, 0].unsqueeze(0)
            pred_y = pred_nodes[:, 1].unsqueeze(0)
            pred_w = weights.unsqueeze(0)

            loss = evaluate_integral(exp_x,exp_y,coeff,
                pred_x, pred_y, pred_w,
                true_x.unsqueeze(0),
                true_y.unsqueeze(0),
                true_w.unsqueeze(0)
            )
            loss.backward()

            # gradient clipping & optimizer step
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for exp_x, exp_y, coeff, true_x, true_y, true_w in val_dataloader:
                exp_x, exp_y, coeff, true_x, true_y, true_w = (
                    x.to(device) for x in (exp_x, exp_y, coeff, true_x, true_y, true_w)
                )

                nodal_vals = preprocessor(
                    exp_x.unsqueeze(0),
                    exp_y.unsqueeze(0),
                    coeff.unsqueeze(0)
                ).squeeze(0)
                node_feat = torch.cat([pos, nodal_vals.unsqueeze(1)], dim=1)
                shifts, weights = model(node_feat, edge_index)

                pred_nodes = pos + shifts
                pred_x = pred_nodes[:, 0].unsqueeze(0)
                pred_y = pred_nodes[:, 1].unsqueeze(0)
                pred_w = weights.unsqueeze(0)

                val_loss += evaluate_integral(exp_x, exp_y, coeff,
                    pred_x, pred_y, pred_w,
                    true_x.unsqueeze(0),
                    true_y.unsqueeze(0),
                    true_w.unsqueeze(0)
                ).item()

        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(f"\nEpoch {epoch + 1}/{epochs}: "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"Time: {epoch_time:.2f}s")

        # save checkpoint periodically
        if (epoch + 1) % save_every == 0:
            save_checkpoint(model, optimizer, epoch + 1, train_loss, checkpoint_path)

    # final model save
    final_model_path = os.path.join(model_dir, "gcnn_model_weights_v3.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Model weights saved: {final_model_path}")

    plt.figure(figsize=(10,5))
    plt.plot(epoch_list, train_losses, label="Training Loss")
    plt.plot(epoch_list, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.title("Training and Validation Loss (Log Scale)")
    plt.savefig(os.path.join(model_dir, "loss_curves.png"), dpi=300)
    plt.show()
    
    avg_epoch_time = np.mean(epoch_times)
    print(f"\nAverage epoch time: {avg_epoch_time:.2f} seconds")
    

    return epoch_list, train_losses, val_losses, epoch_times

if __name__ == "__main__":
    epochs_list, train_losses, val_losses, epoch_times = train_gcnn(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            test_functions,
            epochs,
            checkpoint_path=os.path.join(model_dir, "gcnn_checkpoint.pth"),
            save_every=5
        )
    avg_epoch_time = np.mean(epoch_times)
    print(f"\nAverage epoch time: {avg_epoch_time:.2f} seconds")

