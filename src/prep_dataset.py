# data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
from data_processing import process_file

class SRMRDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        filepath = row['filepath']
        target = row['target']

        # Read the audio file and calculate SRMR
        energy, r = process_file(filepath)
        
        # Convert NumPy array to PyTorch tensor
        energy = torch.tensor(energy, dtype=torch.double)
        return energy, target

