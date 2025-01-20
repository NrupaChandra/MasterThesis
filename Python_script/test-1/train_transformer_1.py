import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import numpy as np
from model_transformer_1 import load_model, save_checkpoint, load_checkpoint
from dataloader_transformer_1 import PolynomialDataset
import utilities

#Custom collate function for padding
def custom_collate_fn(batch):
    exp_x = [item[0] for item in batch]
    exp_y = [item[1] for item in batch]
    coeff = [item[2] for item in batch]
    nodes_x = [item[3] for item in batch]
    nodes_y = [item[4] for item in batch]
    weights = [item[5] for item in batch]

    max_len = max([len(x) for x in nodes_x])
    padded_nodes_x = torch.stack([torch.cat([x, torch.zeros(max_len - len(x))]) for x in nodes_x])
    padded_nodes_y = torch.stack([torch.cat([y, torch.zeros(max_len - len(y))]) for y in nodes_y])
    padded_weights = torch.stack([torch.cat([w, torch.zeros(max_len - len(w))]) for w in weights])
    masks = torch.stack([torch.cat([torch.ones(len(x)), torch.zeros(max_len - len(x))]) for x in nodes_x])

    exp_x = torch.stack(exp_x)
    exp_y = torch.stack(exp_y)
    coeff = torch.stack(coeff)
    return exp_x, exp_y, coeff, padded_nodes_x, padded_nodes_y, padded_weights, masks

# Integration test functions
def test_functions():
    return [
        lambda x,y : 1,
        lambda x,y : x,
        lambda x,y : y,
        lambda x,y : x*y
    ]

    return pred_integral, true_integral

def evaluate_levelset(exp_x, exp_y, coeff, pred_nodes_x, pred_nodes_y):
    values = coeff.unsqueeze(1) * (torch.pow(pred_nodes_x.unsqueeze(2), exp_x.unsqueeze(1)) * torch.pow(pred_nodes_y.unsqueeze(2), exp_y.unsqueeze(1)))
    values = values.sum(dim=2)
    return values

# Combined masked and integration loss
def masked_and_integration_loss(exp_x, exp_y, coeff, pred_nodes_x, pred_nodes_y, pred_weights, 
                                 true_nodes_x, true_nodes_y, true_weights, masks, 
                                 criterion, test_functions):
    # masked_loss = (
    #     criterion(pred_nodes_x * masks, true_nodes_x * masks) +
    #     criterion(pred_nodes_y * masks, true_nodes_y * masks) +
    #     criterion(pred_weights * masks, true_weights * masks)
    # )
    integration_loss = 0
    penalty = 0
    for test_fn in test_functions():
        pred_integral = utilities.compute_integration(pred_nodes_x, pred_nodes_y, pred_weights, test_fn)
        true_integral = utilities.compute_integration(true_nodes_x, true_nodes_y, true_weights, test_fn)
        integration_loss += criterion(pred_integral, true_integral)
    integration_loss /= len(test_functions())

    levelset_values = evaluate_levelset(exp_x, exp_y, coeff, pred_nodes_x, pred_nodes_y)

    penalty += torch.sum(torch.relu(levelset_values))  # Penalize positive values
    penalty /= exp_x.size(0)

    total_loss = integration_loss + penalty
    # return masked_loss + integration_loss
    return total_loss

# Training loop
def train_with_augmented_loss(model, train_dataloader, val_dataloader, optimizer, criterion, test_functions, epochs=10, alpha=1.0, checkpoint_path="transformer_checkpoint.pth", save_every=5):    
    start_epoch = 0
    best_loss = float('inf')

    # Load checkpoint if exists
    try:
        start_epoch, best_loss = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Resuming training from epoch {start_epoch + 1}")
    except FileNotFoundError:
        print("No checkpoint found. Starting fresh training.")



    for epoch in range(start_epoch, epochs):

        # print current learning rate
        for param_group in optimizer.param_groups:
            print(f"Current learning rate: {param_group['lr']:.4e}")

        # training
        model.train()
        train_loss = 0
        for exp_x, exp_y, coeff, true_nodes_x, true_nodes_y, true_weights, masks in train_dataloader:
            exp_x, exp_y, coeff = (exp_x.to(device), exp_y.to(device), coeff.to(device))
            true_nodes_x, true_nodes_y, true_weights, masks = (true_nodes_x.to(device),
                                                               true_nodes_y.to(device),
                                                               true_weights.to(device),
                                                               masks.to(device))
            
            [pred_nodes_x, pred_nodes_y, pred_weights] = model(exp_x, exp_y, coeff, None)

            loss = masked_and_integration_loss(
                exp_x, exp_y, coeff,
                pred_nodes_x, pred_nodes_y, pred_weights,
                true_nodes_x, true_nodes_y, true_weights,
                masks, criterion, test_functions
            )  

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            train_loss += loss.item()

        # validation
        model.eval()
        val_loss = 0
        for exp_x, exp_y, coeff, true_nodes_x, true_nodes_y, true_weights, masks in val_dataloader:
            exp_x, exp_y, coeff = (exp_x.to(device), exp_y.to(device), coeff.to(device))
            true_nodes_x, true_nodes_y, true_weights, masks = (true_nodes_x.to(device),
                                                               true_nodes_y.to(device),
                                                               true_weights.to(device),
                                                               masks.to(device))
            
            [pred_nodes_x, pred_nodes_y, pred_weights] = model(exp_x, exp_y, coeff, None)

            loss = masked_and_integration_loss(
                exp_x, exp_y, coeff,
                pred_nodes_x, pred_nodes_y, pred_weights,
                true_nodes_x, true_nodes_y, true_weights,
                masks, criterion, test_functions
            )  

            val_loss += loss.item()

        train_loss = train_loss / len(train_dataloader)
        val_loss = val_loss / len(val_dataloader)
        
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

        # Save checkpoint every `save_every` epochs
        if (epoch + 1) % save_every == 0 or train_loss < best_loss:
            save_checkpoint(model, optimizer, epoch + 1, train_loss, checkpoint_path)
            if train_loss < best_loss:
                best_loss = train_loss
                print("New best model saved.")

# Set a fixed random seed for reproducibility
seed = 6431
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Main training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)

model = load_model().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Load dataset and dataloader
dataset = PolynomialDataset('TestBernstein_p1_data.txt', 'TestBernstein_p1_output.txt')
# Determine the size of the dataset
dataset_size = len(dataset)
# Define the split sizes (95% training, 5% validation)
train_size = int(0.95 * dataset_size)
val_size = dataset_size - train_size
# Split the dataset into training and validation subsets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(dataset, batch_size=64, collate_fn=custom_collate_fn, shuffle=True)
val_dataloader = DataLoader(dataset, batch_size=64, collate_fn=custom_collate_fn, shuffle=False)

# Train the model
train_with_augmented_loss(model, train_dataloader, val_dataloader, optimizer, criterion, test_functions, epochs=100, alpha=1.0, 
    checkpoint_path="transformer_checkpoint.pth", save_every=5)

# save model
torch.save(model.state_dict(), 'model_transformer_1_weights.pth')