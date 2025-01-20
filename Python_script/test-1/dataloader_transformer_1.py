import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Dataset class
class PolynomialDataset(Dataset):
    def __init__(self, input_file, output_file):
        self.input_data = pd.read_csv(input_file, sep=';')
        self.output_data = pd.read_csv(output_file, sep=';')
    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        exp_x = np.array(list(map(float, self.input_data.iloc[idx]['exp_x'].split(','))))
        exp_y = np.array(list(map(float, self.input_data.iloc[idx]['exp_y'].split(','))))
        coeff = np.array(list(map(float, self.input_data.iloc[idx]['coeff'].split(','))))

        nodes_x = np.array(list(map(float, self.output_data.iloc[idx]['nodes_x'].split(','))))
        nodes_y = np.array(list(map(float, self.output_data.iloc[idx]['nodes_y'].split(','))))
        weights = np.array(list(map(float, self.output_data.iloc[idx]['weights'].split(','))))

        return (torch.tensor(exp_x),
                torch.tensor(exp_y),
                torch.tensor(coeff),
                torch.tensor(nodes_x),
                torch.tensor(nodes_y),
                torch.tensor(weights),
                self.input_data.iloc[idx]['id'])