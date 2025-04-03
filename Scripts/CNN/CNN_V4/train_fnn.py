#!/usr/bin/env python
import torch
import random
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from multidataloader_fnn import MultiChunkDataset  # Your dataset class
from model_cnn import load_shallow_cnn_model, save_checkpoint, load_checkpoint
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Set default dtype to single precision.
torch.set_default_dtype(torch.float32)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}, GPU Name: {torch.cuda.get_device_name(0)}")

# Define paths for data and saving model/checkpoints.
data_dir = "/work/scratch/ng66sume/Root/Data/"
model_dir = "/work/scratch/ng66sume/Models/CNN/CNN_V4/"
os.makedirs(model_dir, exist_ok=True)

#########################
# Custom Collate Function
#########################
def custom_collate_fn(batch):
    """
    Assumes each sample returns:
      exp_x, exp_y, coeff, true_nodes_x, true_nodes_y, true_weights, masks
    """
    exp_x = torch.stack([item[0].float() for item in batch])
    exp_y = torch.stack([item[1].float() for item in batch])
    coeff = torch.stack([item[2].float() for item in batch])
    nodes_x = [item[3].float() for item in batch]
    nodes_y = [item[4].float() for item in batch]
    weights = [item[5].float() for item in batch]
    max_len = max(x.size(0) for x in nodes_x)
    padded_nodes_x = torch.stack([
        torch.cat([x, torch.zeros(max_len - x.size(0), dtype=torch.float32)])
        for x in nodes_x
    ])
    padded_nodes_y = torch.stack([
        torch.cat([y, torch.zeros(max_len - y.size(0), dtype=torch.float32)])
        for y in nodes_y
    ])
    padded_weights = torch.stack([
        torch.cat([w, torch.zeros(max_len - w.size(0), dtype=torch.float32)])
        for w in weights
    ])
    masks = torch.stack([
        torch.cat([torch.ones(x.size(0), dtype=torch.float32),
                   torch.zeros(max_len - x.size(0), dtype=torch.float32)])
        for x in nodes_x
    ])
    return exp_x, exp_y, coeff, padded_nodes_x, padded_nodes_y, padded_weights, masks

#########################
# Integration Loss Function
#########################
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

def integration_loss(pred_nodes_x, pred_nodes_y, pred_weights, 
                     true_nodes_x, true_nodes_y, true_weights, 
                     test_functions):
    """
    Computes the integration loss by comparing the predicted integral (computed
    from predicted nodes and weights) with the true integral (computed from true nodes and weights).
    
    For each test function f(x, y), the integral is computed as a weighted sum:
      integral = sum_i [ weight_i * f(node_x_i, node_y_i) ]
    
    The loss is the average MSE between predicted and true integrals over all test functions.
    """
    total_loss = 0.0
    funcs = test_functions()
    for fn in funcs:
        # Evaluate test function at the node positions.
        pred_vals = fn(pred_nodes_x, pred_nodes_y)  # shape: (batch, num_nodes)
        true_vals = fn(true_nodes_x, true_nodes_y)  # shape: (batch, num_nodes)
        
        pred_integral = torch.sum(pred_weights * pred_vals, dim=1)  # (batch,)
        true_integral = torch.sum(true_weights * true_vals, dim=1)    # (batch,)
        
        total_loss += F.mse_loss(pred_integral, true_integral)
    
    total_loss /= len(funcs)
    return total_loss

#########################
# Training Function
#########################
def train_cnn(model, train_dataloader, val_dataloader, optimizer, criterion, test_functions, 
              num_nodes, domain, epochs=1000, checkpoint_path=None, save_every=5):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(model_dir, "cnn_checkpoint.pth")
    model.to(device)
    train_losses = []
    val_losses = []
    epoch_list = []
    epoch_times = []
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print("Checkpoint found. Loading checkpoint...")
        start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        print("No checkpoint found. Starting training from epoch 1.")
    
    total_steps = epochs * len(train_dataloader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=total_steps,
                                                      pct_start=0.1, anneal_strategy='linear',
                                                      final_div_factor=100, verbose=True)
    
    # Set the fixed predicted grid size.
    pred_grid_size = 16
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        torch.cuda.empty_cache()
        epoch_list.append(epoch + 1)
        model.train()
        train_loss = 0.0
        for batch_idx, (exp_x, exp_y, coeff, true_nodes_x, true_nodes_y, true_weights, masks) in enumerate(train_dataloader):
            exp_x, exp_y, coeff = (x.to(device) for x in (exp_x, exp_y, coeff))
            true_nodes_x, true_nodes_y, true_weights = (x.to(device) for x in (true_nodes_x, true_nodes_y, true_weights))
            optimizer.zero_grad()
            
            # Get predicted weight grid and delta corrections from the model.
            # Now the model returns two outputs: weights and delta corrections.
            pred_weights, delta_out = model(exp_x, exp_y, coeff)  # shapes: (batch, 1, 16, 16) and (batch, 2, 16, 16)
            batch_size = pred_weights.size(0)
            
            # Create a fixed 16x16 base grid for nodes.
            xs_pred = torch.linspace(domain[0], domain[1], pred_grid_size, device=device)
            ys_pred = torch.linspace(domain[0], domain[1], pred_grid_size, device=device)
            X_pred, Y_pred = torch.meshgrid(xs_pred, ys_pred, indexing='ij')
            base_nodes_x = X_pred.flatten().unsqueeze(0).expand(batch_size, -1)  # (batch, 256)
            base_nodes_y = Y_pred.flatten().unsqueeze(0).expand(batch_size, -1)  # (batch, 256)
            
            # Reshape delta corrections: delta_out is (batch, 2, 16, 16), flatten to (batch, 2, 256)
            delta = delta_out.view(batch_size, 2, -1)
            delta_x = delta[:, 0, :]  # (batch, 256)
            delta_y = delta[:, 1, :]  # (batch, 256)
            
            # Compute final corrected node positions.
            pred_nodes_x = base_nodes_x + delta_x
            pred_nodes_y = base_nodes_y + delta_y
            
            # Reshape predicted weights to (batch, 256).
            pred_weights = pred_weights.view(batch_size, -1)
            
            # Compute integration loss.
            loss = integration_loss(pred_nodes_x, pred_nodes_y, pred_weights, 
                                    true_nodes_x, true_nodes_y, true_weights, 
                                    test_functions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (exp_x, exp_y, coeff, true_nodes_x, true_nodes_y, true_weights, masks) in enumerate(val_dataloader):
                exp_x, exp_y, coeff = (x.to(device) for x in (exp_x, exp_y, coeff))
                true_nodes_x, true_nodes_y, true_weights = (x.to(device) for x in (true_nodes_x, true_nodes_y, true_weights))
                pred_weights, delta_out = model(exp_x, exp_y, coeff)
                batch_size = pred_weights.size(0)
                # Create fixed 16x16 base grid for predicted nodes.
                xs_pred = torch.linspace(domain[0], domain[1], pred_grid_size, device=device)
                ys_pred = torch.linspace(domain[0], domain[1], pred_grid_size, device=device)
                X_pred, Y_pred = torch.meshgrid(xs_pred, ys_pred, indexing='ij')
                base_nodes_x = X_pred.flatten().unsqueeze(0).expand(batch_size, -1)
                base_nodes_y = Y_pred.flatten().unsqueeze(0).expand(batch_size, -1)
                
                # Reshape and apply delta corrections.
                delta = delta_out.view(batch_size, 2, -1)
                delta_x = delta[:, 0, :]
                delta_y = delta[:, 1, :]
                pred_nodes_x = base_nodes_x + delta_x
                pred_nodes_y = base_nodes_y + delta_y
                
                # Reshape predicted weights.
                pred_weights = pred_weights.view(batch_size, -1)
                loss = integration_loss(pred_nodes_x, pred_nodes_y, pred_weights, 
                                        true_nodes_x, true_nodes_y, true_weights, 
                                        test_functions)
                val_loss += loss.item()
        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        print(f"\nEpoch {epoch + 1}/{epochs}: Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, Time: {epoch_time:.2f} sec")
        
        if (epoch + 1) % save_every == 0:
            save_checkpoint(model, optimizer, epoch + 1, train_loss, checkpoint_path)
    
    final_model_path = os.path.join(model_dir, 'cnn_model_weights_v3.0.pth')
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

#########################
# Main Training Execution
#########################
if __name__ == "__main__":
    seed = 6432
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset using the base directory.
    dataset = MultiChunkDataset(
        index_file=os.path.join(data_dir, 'preprocessed_chunks_100k/index.txt'),
        base_dir=data_dir
    )
    
    print("Dataset length:", len(dataset))
    if len(dataset) == 0:
        raise ValueError("The dataset is empty. Please verify your index file and data directory.")
    
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4,
        prefetch_factor=4,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1024,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=4,
        prefetch_factor=4,
        pin_memory=True
    )
    
    # Instantiate the model using the updated loader.
    model = load_shallow_cnn_model(num_nodes=1225, domain=(-1,1), dropout_rate=0.0)
    model.to(device)
    
    # Criterion is defined but is used inside integration_loss.
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    epochs = 1000
    epoch_list, train_losses, val_losses, epoch_times = train_cnn(
        model, train_dataloader, val_dataloader,
        optimizer, criterion, test_functions,
        num_nodes=1225, domain=(-1,1),
        epochs=epochs, checkpoint_path=os.path.join(model_dir, "cnn_checkpoint.pth"), save_every=5
    )
    
    avg_epoch_time = np.mean(epoch_times)
    print(f"\nAverage epoch time: {avg_epoch_time:.2f} seconds")
