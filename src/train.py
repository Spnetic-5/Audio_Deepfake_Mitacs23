from __future__ import division
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_processing import calculate_eer
from prep_dataset import SRMRDataset
from crnn_model import initialize_model

BASE_PATH = '/home/sspowar/scratch/archive/LA/LA'

# TRAIN

train_df = pd.read_csv(f'{BASE_PATH}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
                       sep=" ", header=None)
train_df.columns =['speaker_id','filename','system_id','type','class_name']
train_df['filepath'] = f'{BASE_PATH}/ASVspoof2019_LA_train/flac/'+train_df.filename+'.flac'
train_df['target'] = (train_df.class_name=='spoof').astype('int32') # set labels 1 for fake and 0 for real

# Define training parameters
num_epochs = 10
batch_size = 16
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = initialize_model(num_class=2, msr_size=(23, 8), rnn_hidden_size=128, dropout=0.7, tem_fac=[1, 2, 1])
print(model)
model.double()
model.to(device)

# Create DataLoader
train_dataset = SRMRDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    y_true = []
    y_scores = []

    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calculate accuracy
        predicted_labels = (output > 0.5).float()  # Assuming 0.5 threshold for binary classification
        total_correct += (predicted_labels == target).sum().item()
        total_samples += target.size(0)

        # Store true labels and predicted scores for EER calculation
        y_true.extend(target.cpu().numpy())
        y_scores.extend(output.cpu().detach().numpy())

    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples
    eer = calculate_eer(np.array(y_true), np.array(y_scores))
    return avg_loss, accuracy, eer

# Training loop
for epoch in range(1, num_epochs + 1):
    train_loss, train_accuracy, train_eer = train(model, device, train_loader, criterion, optimizer, epoch)
    print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train EER: {train_eer:.4f}")

    # Save the model after each epoch
    if epoch % 2 == 0:
        torch.save(model.state_dict(), f"crnn_model_epoch_{epoch}.pt")
