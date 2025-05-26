#!/usr/bin/env python
import os
import time
import torch
from torch.utils.data import DataLoader
from model_cnn import load_shallow_cnn_model
from multidataloader_fnn import MultiChunkDataset

# Set device and default dtype.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)

# Print the device in use.
print(f"the device in use is: {device}")

# Define directories.
model_dir = '/work/scratch/ng66sume/Models/CNN/CNN_V1/'
data_dir = '/work/scratch/ng66sume/Root/Data/'

# Load the model checkpoint and set to evaluation mode.
model_path = os.path.join(model_dir, 'cnn_model_weights_v1.0.pth')
model = load_shallow_cnn_model(model_path).to(device)
model.eval()

# Prepare the dataset and dataloader.
dataset = MultiChunkDataset(
    index_file=os.path.join(data_dir, 'preprocessed_chunks_3_10k/index.txt'),
    base_dir=data_dir
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Measure the total inference time for all samples.
with torch.no_grad():
    # Synchronize the GPU (if available) before starting.
    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_start = time.time()
    
    for sample in dataloader:
        # Each sample is assumed to be a tuple: (exp_x, exp_y, coeff, sample_id)
        exp_x, exp_y, coeff, sample_id = sample

        # Move inputs to device.
        exp_x = exp_x.to(device, dtype=torch.float32, non_blocking=True)
        exp_y = exp_y.to(device, dtype=torch.float32, non_blocking=True)
        coeff = coeff.to(device, dtype=torch.float32, non_blocking=True)

        _ = model(exp_x, exp_y, coeff)  # Run inference

        # Ensure GPU processes are finished before moving on.
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    # Synchronize again after the loop.
    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_end = time.time()

total_time = total_end - total_start
print(f"Total inference time for all samples: {total_time:.12f} seconds")
