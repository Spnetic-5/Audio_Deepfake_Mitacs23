from __future__ import division
import numpy as np
import pandas as pd
from scipy.signal import hamming
from srmrpy.hilbert import hilbert
from srmrpy.modulation_filters import *
from gammatone.fftweight import fft_gtgram
from gammatone.filters import centre_freqs, make_erb_filters, erb_filterbank
from srmrpy.segmentaxis import segment_axis
from scipy.io.wavfile import read as readwav
import soundfile as sf
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, roc_auc_score

BASE_PATH = '/home/sspowar/scratch/archive/LA/LA'

# TRAIN

train_df = pd.read_csv(f'{BASE_PATH}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
                       sep=" ", header=None)
train_df.columns =['speaker_id','filename','system_id','type','class_name']
# train_df.drop(columns=['null'],inplace=True)
train_df['filepath'] = f'{BASE_PATH}/ASVspoof2019_LA_train/flac/'+train_df.filename+'.flac'
train_df['target'] = (train_df.class_name=='spoof').astype('int32') # set labels 1 for fake and 0 for real


def calc_erbs(low_freq, fs, n_filters):
    ear_q = 9.26449 # Glasberg and Moore Parameters
    min_bw = 24.7
    order = 1

    erbs = ((centre_freqs(fs, n_filters, low_freq)/ear_q)**order + min_bw**order)**(1/order)
    return erbs

def calc_cutoffs(cfs, fs, q):
    # Calculates cutoff frequencies (3 dB) for 2nd order bandpass
    w0 = 2*np.pi*cfs/fs
    B0 = np.tan(w0/2)/q
    L = cfs - (B0 * fs / (2*np.pi))
    R = cfs + (B0 * fs / (2*np.pi))
    return L, R

def normalize_energy(energy, drange=30.0):
    peak_energy = np.max(np.mean(energy, axis=0))
    min_energy = peak_energy*10.0**(-drange/10.0)
    energy[energy < min_energy] = min_energy
    energy[energy > peak_energy] = peak_energy
    return energy

def srmr(x, fs, n_cochlear_filters=23, low_freq=125, min_cf=4, max_cf=128, fast=True, norm=False):
    wLengthS = .256
    wIncS = .064
    # Computing gammatone envelopes
    if fast:
        mfs = 400.0
        gt_env = fft_gtgram(x, fs, 0.010, 0.0025, n_cochlear_filters, low_freq)
    else:
        cfs = centre_freqs(fs, n_cochlear_filters, low_freq)
        fcoefs = make_erb_filters(fs, cfs)
        gt_env = np.abs(hilbert(erb_filterbank(x, fcoefs)))
        mfs = fs

    wLength = int(np.ceil(wLengthS*mfs))
    wInc = int(np.ceil(wIncS*mfs))

    # Computing modulation filterbank with Q = 2 and 8 channels
    mod_filter_cfs = compute_modulation_cfs(min_cf, max_cf, 8)
    MF = modulation_filterbank(mod_filter_cfs, mfs, 2)

    n_frames = int(1 + (gt_env.shape[1] - wLength)//wInc)
    w = hamming(wLength+1)[:-1] # window is periodic, not symmetric

    energy = np.zeros((n_cochlear_filters, 8, n_frames))
    for i, ac_ch in enumerate(gt_env):
        mod_out = modfilt(MF, ac_ch)
        for j, mod_ch in enumerate(mod_out):
            mod_out_frame = segment_axis(mod_ch, wLength, overlap=wLength-wInc, end='pad')
            energy[i,j,:] = np.sum((w*mod_out_frame[:n_frames])**2, axis=1)

    if norm:
        energy = normalize_energy(energy)

    erbs = np.flipud(calc_erbs(low_freq, fs, n_cochlear_filters))

    avg_energy = np.mean(energy, axis=2)
    total_energy = np.sum(avg_energy)

    AC_energy = np.sum(avg_energy, axis=1)
    AC_perc = AC_energy*100/total_energy

    AC_perc_cumsum=np.cumsum(np.flipud(AC_perc))
    K90perc_idx = np.where(AC_perc_cumsum>90)[0][0]

    BW = erbs[K90perc_idx]

    cutoffs = calc_cutoffs(mod_filter_cfs, fs, 2)[0]

    if (BW > cutoffs[4]) and (BW < cutoffs[5]):
        Kstar=5
    elif (BW > cutoffs[5]) and (BW < cutoffs[6]):
        Kstar=6
    elif (BW > cutoffs[6]) and (BW < cutoffs[7]):
        Kstar=7
    elif (BW > cutoffs[7]):
        Kstar=8

    return np.sum(avg_energy[:, :4])/np.sum(avg_energy[:, 4:Kstar]), energy


def read_audio_file(filename, max_length=64600):
    # Use soundfile to read the .flac audio file
    audio_data, fs = sf.read(filename, always_2d=True, dtype='float32')
    # Normalize the audio data to the range [-1, 1]
    audio_data /= np.max(np.abs(audio_data))

    # Pad or truncate audio data to the desired max_length
    if len(audio_data) < max_length:
        padding = max_length - len(audio_data)
        audio_data = np.pad(audio_data, ((0, padding), (0, 0)), mode='constant')
    elif len(audio_data) > max_length:
        audio_data = audio_data[:max_length, :]
    return fs, audio_data

def process_file(f, n_cochlear_filters=23, low_freq=125, min_cf=4, max_cf=128, fast=True, norm=True):
    fs, s = read_audio_file(f)  # Use read_audio_file to handle .flac files
    if len(s.shape) > 1:
        s = s[:, 0]
    r, energy = srmr(s, fs, n_cochlear_filters=n_cochlear_filters,
                     min_cf=min_cf,
                     max_cf=max_cf,
                     fast=fast,
                     norm=norm)
    return energy, r



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


def calculate_eer(y_true, y_scores):
    fpr, fnr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - fnr)))]
    return eer

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


class CRNN_Model(nn.Module):
    def __init__(self, num_class:int, msr_size:tuple, rnn_hidden_size:int, dropout:float, tem_fac:list):
        super(CRNN_Model, self).__init__()
        self.num_class = num_class
        self.rnn_hidden_size = rnn_hidden_size
        self.dp = nn.Dropout(p=dropout)
        self.num_freq = msr_size[0]
        self.num_mod = msr_size[1]
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.tem_fac = tem_fac
        
        self.cnn1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.MaxPool3d((self.tem_fac[0], 1, 1)),
            self.relu
        )
        
        self.cnn2 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.MaxPool3d((self.tem_fac[1], 1, 1)),
            self.relu
        )
        
        self.cnn3 = nn.Sequential(
            nn.Conv3d(16, 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(4),
            nn.MaxPool3d((self.tem_fac[2], 1, 1)),
            self.relu
        )
        
        
        self.downsample = nn.MaxPool3d((2,2,2))
        
        self.CNNblock = nn.Sequential(
            self.cnn1,
            self.cnn2,
            self.cnn3
            )
        
        # self.Att = CBAM_Att(channel=4)
        
        self.fc1 = nn.Sequential(
            nn.Linear(20416, 128),
            nn.BatchNorm1d(128),
            self.relu,
            self.dp
            )
        
        # RNN
        self.rnn1 = nn.GRU(input_size=128, 
                            hidden_size=self.rnn_hidden_size,
                            num_layers=3,
                            bidirectional=True, 
                            batch_first=True)
        
        self.layer_norm = nn.LayerNorm([2*self.rnn_hidden_size,int(150/np.product(self.tem_fac))])
        self.maxpool = nn.MaxPool1d(int(150/np.product(self.tem_fac)))
        
        self.fc2 = nn.Linear(self.rnn_hidden_size*2, 1)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
    
    def forward(self, x):
        # Apply the CNN layers
        # print(x.shape)
        x = x.unsqueeze(1) 
        # print(x.shape)
        ot = self.CNNblock(x)
        # Flatten ot to have shape (batch_size, -1)
        ot = ot.view(ot.size(0), -1)

        # print(ot.shape)
        # Pass through the first fully connected layer
        ot = self.fc1(ot)
        # print(ot.shape)   
        # After the RNN layer, ot will have shape (batch_size, time_steps, rnn_hidden_size * 2)
        ot, _ = self.rnn1(ot)
        # print("RNN", ot.shape)
        
        ot = self.fc2(ot).squeeze(1).float()
        ot = torch.sigmoid(ot)
        # print("After FC2", ot.shape, ot)
        
        return ot


train_dataset = SRMRDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = CRNN_Model(num_class=2, msr_size=(23, 8), rnn_hidden_size=128, dropout=0.7, tem_fac=[1, 2, 1])
print(model)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.double()
model.to(device)



# Train the model for a few epochs
num_epochs = 10

for epoch in range(1, num_epochs + 1):
    train_loss, train_accuracy, train_eer = train(model, device, train_loader, criterion, optimizer, epoch)
    print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train EER: {train_eer:.4f}")

    # Save the model after each epoch
    if epoch % 2 == 0:
        torch.save(model.state_dict(), f"crnn_model_epoch_{epoch}.pt")



# TEST

# model = CRNN_Model(num_class=2, msr_size=(23, 8), rnn_hidden_size=128, dropout=0.7, tem_fac=[1, 2, 1])
# model.load_state_dict(torch.load("crnn_model_epoch_10.pt", map_location=torch.device('cpu')))  # Replace X with the desired epoch number
# model.eval()  # Set the model to evaluation mode
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.double()
# model.to(device)
        
# test_df = pd.read_csv(f'{BASE_PATH}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt',
#                        sep=" ", header=None)
# test_df.columns =['speaker_id','filename','system_id','null','class_name']
# test_df.drop(columns=['null'],inplace=True)
# test_df['filepath'] = f'{BASE_PATH}/ASVspoof2019_LA_dev/flac/'+test_df.filename+'.flac'
# test_df['target'] = (test_df.class_name=='spoof').astype('int32')

# test_dataset = SRMRDataset(test_df)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# total_correct = 0
# total_samples = 0

# # Create a tqdm progress bar for the test_loader
# with tqdm(test_loader, desc="Testing") as pbar:
#     for data, target in pbar:
#         data, target = data.to(device), target.to(device)
#         with torch.no_grad():
#             output = model(data)
#         predicted_labels = (output > 0.5).float()  # Assuming 0.5 threshold for binary classification
#         total_correct += (predicted_labels == target).sum().item()
#         total_samples += target.size(0)

# test_accuracy = total_correct / total_samples
# print(f"Test Accuracy: {test_accuracy:.4f}")

