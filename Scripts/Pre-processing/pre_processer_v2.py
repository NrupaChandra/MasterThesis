#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import torch

def _parse_value(value):
    """
    Helper function to parse a comma-separated string into a NumPy array.
    """
    return np.array(list(map(float, value.split(','))), dtype=np.float32)

def pad_fixed(arr, fixed_length=16):
    """
    Pads or truncates a 1D NumPy array to a fixed length.
    
    Parameters:
      - arr: Input 1D NumPy array.
      - fixed_length: The desired fixed length.
    
    Returns:
      A 1D NumPy array of length fixed_length.
    """
    current_length = len(arr)
    if current_length < fixed_length:
        pad_width = fixed_length - current_length
        return np.pad(arr, (0, pad_width), mode='constant', constant_values=0)
    else:
        return arr[:fixed_length]

def chain_csv(files, **kwargs):
    """
    Generator that chains together chunks from multiple CSV files.
    
    Parameters:
      - files: A list of file paths.
      - kwargs: Additional keyword arguments to pass to pd.read_csv.
    Yields:
      Chunks (DataFrames) from each file in sequence.
    """
    for file in files:
        print(f"Reading from {file}...")
        for chunk in pd.read_csv(file, **kwargs):
            yield chunk

def preprocess_data(data_files, output_files, save_dir='combined_preprocessed_chunks_10kBernstein', chunksize=50000):
    """
    Combines multiple data and output files, processes them in chunks, and saves the preprocessed data.
    
    Parameters:
      - data_files: List of paths to the raw input (data) files.
      - output_files: List of paths to the raw output files.
      - save_dir: Directory where the processed chunk files will be saved.
      - chunksize: Number of rows to process per chunk.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create generators to yield chunks from the combined files.
    data_reader = chain_csv(data_files, sep=';', chunksize=chunksize)
    output_reader = chain_csv(output_files, sep=';', chunksize=chunksize)
    
    chunk_idx = 0
    chunk_files = []
    
    # Process the combined chunks
    for data_chunk, output_chunk in zip(data_reader, output_reader):
        data_list = []
        for idx in range(len(data_chunk)):
            # Parse and ensure fixed size for exp_x, exp_y, and coeff
            exp_x   = pad_fixed(_parse_value(data_chunk.iloc[idx]['exp_x']), fixed_length=16)
            exp_y   = pad_fixed(_parse_value(data_chunk.iloc[idx]['exp_y']), fixed_length=16)
            coeff   = pad_fixed(_parse_value(data_chunk.iloc[idx]['coeff']), fixed_length=16)
            
            nodes_x = _parse_value(output_chunk.iloc[idx]['nodes_x'])
            nodes_y = _parse_value(output_chunk.iloc[idx]['nodes_y'])
            weights = _parse_value(output_chunk.iloc[idx]['weights'])
            sample_id = data_chunk.iloc[idx]['id']
            
            # Convert the numpy arrays to torch tensors.
            sample = (
                torch.tensor(exp_x, dtype=torch.float32),
                torch.tensor(exp_y, dtype=torch.float32),
                torch.tensor(coeff, dtype=torch.float32),
                torch.tensor(nodes_x, dtype=torch.float32),
                torch.tensor(nodes_y, dtype=torch.float32),
                torch.tensor(weights, dtype=torch.float32),
                sample_id  # Keeping the ID as is
            )
            data_list.append(sample)
        
        # Save the processed chunk to a file.
        chunk_file = os.path.join(save_dir, f'preprocessed_chunk{chunk_idx}.pt')
        torch.save(data_list, chunk_file)
        chunk_files.append(chunk_file)
        print(f"Saved chunk {chunk_idx} with {len(data_list)} samples to {chunk_file}")
        chunk_idx += 1
    
    # Create an index file listing all chunk files.
    index_file = os.path.join(save_dir, 'index.txt')
    with open(index_file, 'w') as f:
        for cf in chunk_files:
            f.write(cf + "\n")
    print(f"Index file saved to {index_file}")

if __name__ == "__main__":
    # Lists of data and output files to be combined.
    data_files = [
        "10kBernstein_p1_data.txt", 
        "10kBernstein_p2_data.txt", 
        "10kBernstein_p3_data.txt"
    ]
    output_files = [
        "10kBernstein_p1_output_8.txt", 
        "10kBernstein_p2_output_8.txt", 
        "10kBernstein_p3_output_8.txt"
    ]
    
    preprocess_data(data_files, output_files)
