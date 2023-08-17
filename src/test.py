"""
@author: Saurabh.Powar
"""

import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from prep_dataset import SRMRDataset
from crnn_model import CRNN_Model

BASE_PATH = '/home/sspowar/scratch/archive/LA/LA'

# Load saved model
model = CRNN_Model(num_class=2, msr_size=(23, 8), rnn_hidden_size=128, dropout=0.7, tem_fac=[1, 2, 1])
model.load_state_dict(torch.load("crnn_model_epoch_10.pt", map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.double()
model.to(device)

# Load test data
test_df = pd.read_csv(f'{BASE_PATH}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt',
                       sep=" ", header=None)
test_df.columns =['speaker_id','filename','system_id','null','class_name']
test_df.drop(columns=['null'], inplace=True)
test_df['filepath'] = f'{BASE_PATH}/ASVspoof2019_LA_dev/flac/'+test_df.filename+'.flac'
test_df['target'] = (test_df.class_name=='spoof').astype('int32')

test_dataset = SRMRDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

total_correct = 0
total_samples = 0

# Testing loop
with tqdm(test_loader, desc="Testing") as pbar:
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
        predicted_labels = (output > 0.5).float()  # Assuming 0.5 threshold for binary classification
        total_correct += (predicted_labels == target).sum().item()
        total_samples += target.size(0)

test_accuracy = total_correct / total_samples
print(f"Test Accuracy: {test_accuracy:.4f}")
