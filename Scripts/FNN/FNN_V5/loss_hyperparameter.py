#!/usr/bin/env python
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from multidataloader_fnn import MultiChunkDataset  # Your dataset class
from model_fnn import load_ff_pipelines_model, save_checkpoint, load_checkpoint
import optuna.visualization as vis

optuna.logging.set_verbosity(optuna.logging.DEBUG)

# Set default dtype to single precision.
torch.set_default_dtype(torch.float32)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}, GPU Name: {torch.cuda.get_device_name(0)}")

# Define paths for data and saving model/checkpoints.
data_dir = "/work/scratch/ng66sume/Root/Data/"
model_dir = "/work/home/ng66sume/MasterThesis/Models/FNN_model_v5/"
os.makedirs(model_dir, exist_ok=True)

#########################
# Custom Collate Function
#########################
def custom_collate_fn(batch):
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
# Integration Loss Functions
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

def evaluate_levelset(exp_x, exp_y, coeff, pred_nodes_x, pred_nodes_y):
    values = coeff.unsqueeze(1) * (torch.pow(pred_nodes_x.unsqueeze(2), exp_x.unsqueeze(1)) *
                                   torch.pow(pred_nodes_y.unsqueeze(2), exp_y.unsqueeze(1)))
    return values.sum(dim=2)

def separation_loss_2d(pred_nodes_x, pred_nodes_y, threshold_factor=0.1):
    if pred_nodes_x.shape[1] < 2:
        return torch.tensor(0.0, device=pred_nodes_x.device)
    diff_x = pred_nodes_x[:, 1:] - pred_nodes_x[:, :-1]
    diff_y = pred_nodes_y[:, 1:] - pred_nodes_y[:, :-1]
    squared_sum = diff_x ** 2 + diff_y ** 2
    euclidean_distances = torch.sqrt(torch.clamp(squared_sum, min=1e-12))
    batch_size = pred_nodes_x.shape[0]
    thresholds = []
    for i in range(batch_size):
        nodes = torch.stack([pred_nodes_x[i], pred_nodes_y[i]], dim=1)
        diff_nodes = nodes.unsqueeze(0) - nodes.unsqueeze(1)
        dists = torch.sqrt(torch.clamp(torch.sum(diff_nodes ** 2, dim=2), min=1e-12))
        max_dist = dists.max()
        dynamic_threshold = threshold_factor * max_dist
        thresholds.append(dynamic_threshold)
    thresholds = torch.stack(thresholds).unsqueeze(1).expand_as(euclidean_distances)
    penalty = torch.clamp(thresholds - euclidean_distances, min=0)
    return penalty.mean()

def masked_and_integration_loss(exp_x, exp_y, coeff, pred_nodes_x, pred_nodes_y, pred_weights, 
                                true_nodes_x, true_nodes_y, true_weights, masks, 
                                criterion, test_functions,
                                w_integration=1.0, w_levelset=1.0, w_separation=1.0, threshold_factor=0.1):
    # Integration loss computed over a series of test functions.
    integration_loss = 0
    for test_fn in test_functions():
        pred_integral = torch.sum(pred_weights * test_fn(pred_nodes_x, pred_nodes_y), dim=1)
        true_integral = torch.sum(true_weights * test_fn(true_nodes_x, true_nodes_y), dim=1)
        integration_loss += criterion(pred_integral, true_integral)
    integration_loss /= len(test_functions())
    
    # Levelset penalty.
    levelset_values = evaluate_levelset(exp_x, exp_y, coeff, pred_nodes_x, pred_nodes_y)
    levelset_penalty = torch.sum(torch.relu(levelset_values)) / exp_x.size(0)
    
    # Separation penalty with dynamic threshold.
    separation_penalty = separation_loss_2d(pred_nodes_x, pred_nodes_y, threshold_factor)
    
    total_loss = (w_integration * integration_loss + 
                  w_levelset * levelset_penalty + 
                  w_separation * separation_penalty)
    return total_loss

#########################
# Training Function
#########################
def train_fnn(model, train_dataloader, val_dataloader, optimizer, criterion, test_functions, 
              epochs=1000, checkpoint_path=None, save_every=5,
              loss_weights=None, threshold_factor=0.1):
    # loss_weights should be a dict with keys: w_integration, w_levelset, w_separation.
    if loss_weights is None:
        loss_weights = {"w_integration": 1.0, "w_levelset": 1.0, "w_separation": 1.0}
    if checkpoint_path is None:
        checkpoint_path = os.path.join(model_dir, "fnn_checkpoint.pth")
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
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        torch.cuda.empty_cache()
        epoch_list.append(epoch + 1)
        model.train()
        train_loss = 0
        for batch_idx, (exp_x, exp_y, coeff, true_nodes_x, true_nodes_y, true_weights, masks) in enumerate(train_dataloader):
            exp_x, exp_y, coeff, true_nodes_x, true_nodes_y, true_weights, masks = (
                x.to(device) for x in (exp_x, exp_y, coeff, true_nodes_x, true_nodes_y, true_weights, masks)
            )
            optimizer.zero_grad()
            pred_nodes_x, pred_nodes_y, pred_weights = model(exp_x, exp_y, coeff)
            loss = masked_and_integration_loss(exp_x, exp_y, coeff,
                                               pred_nodes_x, pred_nodes_y, pred_weights,
                                               true_nodes_x, true_nodes_y, true_weights,
                                               masks, criterion, test_functions,
                                               loss_weights["w_integration"],
                                               loss_weights["w_levelset"],
                                               loss_weights["w_separation"],
                                               threshold_factor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (exp_x, exp_y, coeff, true_nodes_x, true_nodes_y, true_weights, masks) in enumerate(val_dataloader):
                exp_x, exp_y, coeff, true_nodes_x, true_nodes_y, true_weights, masks = (
                    x.to(device) for x in (exp_x, exp_y, coeff, true_nodes_x, true_nodes_y, true_weights, masks)
                )
                pred_nodes_x, pred_nodes_y, pred_weights = model(exp_x, exp_y, coeff)
                loss = masked_and_integration_loss(exp_x, exp_y, coeff,
                                                   pred_nodes_x, pred_nodes_y, pred_weights,
                                                   true_nodes_x, true_nodes_y, true_weights,
                                                   masks, criterion, test_functions,
                                                   loss_weights["w_integration"],
                                                   loss_weights["w_levelset"],
                                                   loss_weights["w_separation"],
                                                   threshold_factor)
                val_loss += loss.item()
        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        print(f"\nEpoch {epoch + 1}/{epochs}: Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, Time: {epoch_time:.2f} sec")
        
        if (epoch + 1) % save_every == 0:
            save_checkpoint(model, optimizer, epoch + 1, train_loss, checkpoint_path)
    
    final_model_path = os.path.join(model_dir, 'fnn_model_weights_v5.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Model weights saved: {final_model_path}")
    
    return epoch_list, train_losses, val_losses, epoch_times

#########################
# Helper: Create Dataloaders
#########################
def get_dataloaders(batch_size):
    dataset = MultiChunkDataset(
        index_file=os.path.join(data_dir, 'preprocessed_chunks_5k/index.txt'),
        base_dir=data_dir
    )
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4,
        prefetch_factor=4,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=4,
        prefetch_factor=4,
        pin_memory=True
    )
    return train_dataloader, val_dataloader

#########################
# Optuna Objective Function for Loss Hyperparameter Study
#########################
def objective_loss_hyperparams(trial):
    print(f"Starting loss hyperparameter trial {trial.number}")
    # Use the best model hyperparameters from your previous study.

    # Sample loss hyperparameters.
    w_integration = trial.suggest_uniform("w_integration", 0, 1.0)
    w_levelset = trial.suggest_uniform("w_levelset", 0, 1.0)
    w_separation = trial.suggest_uniform("w_separation", 0, 1.0)
    threshold_factor = trial.suggest_uniform("threshold_factor", 0, 0.2)
     
    # Load model with best fixed hyperparameters.
    model = load_ff_pipelines_model(
        weights_path=None,
        hidden_dim=1024,
        output_dim=1024,
        max_output_len=16,
        num_nodes=25,
        domain=(-1,1),
        dropout_rate=0.2493195279014973,
        num_shared_layers=3
    )
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.002360175691100973, weight_decay=0.0058148485830287495)
    criterion = nn.MSELoss()
    
    # Use a fixed batch size for this study.
    batch_size = 256
    train_dataloader, val_dataloader = get_dataloaders(batch_size)
    
    # Use fewer epochs for the hyperparameter study.
    epochs = 200
    for epoch in range(epochs):
        model.train()
        for batch in train_dataloader:
            exp_x, exp_y, coeff, true_nodes_x, true_nodes_y, true_weights, masks = batch
            exp_x, exp_y, coeff = exp_x.to(device), exp_y.to(device), coeff.to(device)
            true_nodes_x, true_nodes_y, true_weights, masks = (
                true_nodes_x.to(device), true_nodes_y.to(device), true_weights.to(device), masks.to(device)
            )
            optimizer.zero_grad()
            pred_nodes_x, pred_nodes_y, pred_weights = model(exp_x, exp_y, coeff)
            loss = masked_and_integration_loss(exp_x, exp_y, coeff,
                                               pred_nodes_x, pred_nodes_y, pred_weights,
                                               true_nodes_x, true_nodes_y, true_weights,
                                               masks, criterion, test_functions,
                                               w_integration, w_levelset, w_separation,
                                               threshold_factor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Evaluate on validation set.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                exp_x, exp_y, coeff, true_nodes_x, true_nodes_y, true_weights, masks = batch
                exp_x, exp_y, coeff = exp_x.to(device), exp_y.to(device), coeff.to(device)
                true_nodes_x, true_nodes_y, true_weights, masks = (
                    true_nodes_x.to(device), true_nodes_y.to(device), true_weights.to(device), masks.to(device)
                )
                pred_nodes_x, pred_nodes_y, pred_weights = model(exp_x, exp_y, coeff)
                loss = masked_and_integration_loss(exp_x, exp_y, coeff,
                                                   pred_nodes_x, pred_nodes_y, pred_weights,
                                                   true_nodes_x, true_nodes_y, true_weights,
                                                   masks, criterion, test_functions,
                                                   w_integration, w_levelset, w_separation,
                                                   threshold_factor)
                val_loss += loss.item()
            val_loss /= len(val_dataloader)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            print(f"Trial {trial.number} pruned with parameters: {trial.params} at epoch {epoch} with val_loss {val_loss}")
            raise optuna.exceptions.TrialPruned()
    
    print(f"Trial {trial.number} finished with parameters: {trial.params} and final val_loss: {val_loss}")
    return val_loss

#########################
# Main Execution: Run Loss Hyperparameter Study
#########################
if __name__ == "__main__":
    seed = 6432
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_loss_hyperparams, n_trials=25)
    print("Best loss hyperparameters: ", study.best_params)
    
    # Generate and save the plots.

    fig2 = vis.plot_param_importances(study)
    fig2.write_image("param_importances_loss.png")
