#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import torch

def _parse_value(value):
    return np.array(list(map(float, value.split(','))), dtype=np.float32)

def preprocess_data(input_file, save_dir=r'C:\Git\Data\preprocessed_chunks_3_10k', chunksize=50000):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create a reader for the input file with the specified chunksize.
    input_reader = pd.read_csv(input_file, sep=';', chunksize=chunksize)
    
    chunk_idx = 0
    chunk_files = []
    
    # Process the file chunk by chunk.
    for input_chunk in input_reader:
        data_list = []
        # Iterate over the rows in the current chunk.
        for idx in range(len(input_chunk)):
            exp_x   = _parse_value(input_chunk.iloc[idx]['exp_x'])
            exp_y   = _parse_value(input_chunk.iloc[idx]['exp_y'])
            coeff   = _parse_value(input_chunk.iloc[idx]['coeff'])
            sample_id = input_chunk.iloc[idx]['id']
            
            # Convert the numpy arrays to torch tensors.
            sample = (
                torch.tensor(exp_x, dtype=torch.float32),
                torch.tensor(exp_y, dtype=torch.float32),
                torch.tensor(coeff, dtype=torch.float32),
                sample_id  # Keeping the ID as is 
            )
            data_list.append(sample)
        
        # Save the processed chunk to a separate file.
        chunk_file = os.path.join(save_dir, f'preprocessed_chunk{chunk_idx}.pt')
        torch.save(data_list, chunk_file)
        chunk_files.append(chunk_file)
        print(f"Saved chunk {chunk_idx} with {len(data_list)} samples to {chunk_file}")
        chunk_idx += 1
    
    # Save an index file listing all chunk files.
    index_file = os.path.join(save_dir, 'index.txt')
    with open(index_file, 'w') as f:
        for chunk_file in chunk_files:
            f.write(chunk_file + "\n")
    print(f"Index file saved to {index_file}")

if __name__ == "__main__":
    input_file = r'C:\Git\Data\10kBernstein_p3_data.txt'
    preprocess_data(input_file)
