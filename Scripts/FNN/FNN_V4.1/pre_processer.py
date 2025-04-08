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

def preprocess_data(input_file, output_file, save_dir='preprocessed_chunks_TestBernstein_4', chunksize=50000):
    """
    Reads the raw input and output text files in chunks, preprocesses each sample,
    and saves the preprocessed data for each chunk into separate binary files.
    
    Parameters:
      - input_file: Path to the raw input file.
      - output_file: Path to the raw output file.
      - save_dir: Directory where the processed chunks will be saved.
      - chunksize: Number of rows to process per chunk.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create readers for both files with the specified chunksize.
    input_reader = pd.read_csv(input_file, sep=';', chunksize=chunksize)
    output_reader = pd.read_csv(output_file, sep=';', chunksize=chunksize)
    
    chunk_idx = 0
    chunk_files = []
    
    # Process both files chunk by chunk in parallel.
    for input_chunk, output_chunk in zip(input_reader, output_reader):
        data_list = []
        # Iterate over the rows in the current chunk.
        for idx in range(len(input_chunk)):
            exp_x   = _parse_value(input_chunk.iloc[idx]['exp_x'])
            exp_y   = _parse_value(input_chunk.iloc[idx]['exp_y'])
            coeff   = _parse_value(input_chunk.iloc[idx]['coeff'])
            
            nodes_x = _parse_value(output_chunk.iloc[idx]['nodes_x'])
            nodes_y = _parse_value(output_chunk.iloc[idx]['nodes_y'])
            weights = _parse_value(output_chunk.iloc[idx]['weights'])
            sample_id = input_chunk.iloc[idx]['id']
            
            # Convert the numpy arrays to torch tensors.
            sample = (
                torch.tensor(exp_x, dtype=torch.float32),
                torch.tensor(exp_y, dtype=torch.float32),
                torch.tensor(coeff, dtype=torch.float32),
                torch.tensor(nodes_x, dtype=torch.float32),
                torch.tensor(nodes_y, dtype=torch.float32),
                torch.tensor(weights, dtype=torch.float32),
                sample_id  # Keeping the ID as is (or convert if needed)
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
    # Replace these with the paths to your input and output text files.
    input_file  = 'TestBernstein_p1_data.txt'
    output_file = 'testBernstein_p1_output_4.txt'
    preprocess_data(input_file, output_file)
