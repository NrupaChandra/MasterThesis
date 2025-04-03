#!/usr/bin/env python
import os
import torch
from torch.utils.data import Dataset

class MultiChunkDataset(Dataset):
    """
    A dataset class that loads all preprocessed data chunks into memory during initialization.
    This approach assumes that the entire dataset can fit in memory.
    """

    def __init__(self, index_file, base_dir=None):
        """
        Initializes the dataset by reading the index file containing chunk paths and loading all data.

        Parameters:
          - index_file (str): Path to the index file that lists all chunk file paths.
          - base_dir (str, optional): Base directory to prepend to each chunk file path if the path is relative.
        """
        # Read chunk file paths from the index file.
        with open(index_file, 'r') as f:
            chunk_files = [line.strip() for line in f.readlines() if line.strip()]

        # If a base directory is provided, prepend it to relative paths.
        if base_dir is not None:
            self.chunk_files = [
                os.path.join(base_dir, file) if not os.path.isabs(file) else file
                for file in chunk_files
            ]
        else:
            self.chunk_files = chunk_files

        # Load all chunks into memory.
        self.data = []
        for chunk_file in self.chunk_files:
            # Debug: Print which file is being loaded.
            # print(f"Loading chunk: {chunk_file}")
            chunk_data = torch.load(chunk_file)
            self.data.extend(chunk_data)  # Append all samples from the chunk

        print(f"Loaded {len(self.data)} samples from {len(self.chunk_files)} chunks.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
