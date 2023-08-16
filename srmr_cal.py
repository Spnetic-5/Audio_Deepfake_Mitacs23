from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hamming
from srmrpy.hilbert import hilbert
from srmrpy.modulation_filters import *
from gammatone.fftweight import fft_gtgram
from gammatone.filters import centre_freqs, make_erb_filters, erb_filterbank
from srmrpy.segmentaxis import segment_axis
from scipy.io.wavfile import read as readwav
import soundfile as sf
from tqdm import tqdm

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

def read_audio_file(filename):
    # Use soundfile to read the .flac audio file
    audio_data, fs = sf.read(filename, always_2d=True, dtype='float32')
    # Normalize the audio data to the range [-1, 1]
    audio_data /= np.max(np.abs(audio_data))
    return fs, audio_data

def process_file(f, n_cochlear_filters=23, low_freq=125, min_cf=4, max_cf=128, fast=True, norm=False):
    fs, s = read_audio_file(f)  # Use read_audio_file to handle .flac files
    if len(s.shape) > 1:
        s = s[:, 0]
    r, energy = srmr(s, fs, n_cochlear_filters=n_cochlear_filters,
                     min_cf=min_cf,
                     max_cf=max_cf,
                     fast=fast,
                     norm=norm)
    return energy, r

def plot_srmr(average_energy_real, average_energy_fake, save_path=None):
    # Plot the average energy for each cochlear filter and frame for both real and fake recordings
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Average SRMR for Real Recordings')
    sns.heatmap(average_energy_real.T, cmap='viridis', xticklabels=10, yticklabels=10)
    plt.xlabel('Cochlear Filters')
    plt.ylabel('Frames')
    
    plt.subplot(1, 2, 2)
    plt.title('Average SRMR for Fake Recordings')
    sns.heatmap(average_energy_fake.T, cmap='viridis', xticklabels=10, yticklabels=10)
    plt.xlabel('Cochlear Filters')
    plt.ylabel('Frames')
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
        
    plt.show()


def calculate_average_srmr(recordings):
    print("Calculating ...")
    total_energy = None
    n_recordings = len(recordings)
    
    for recording in tqdm(recordings, desc="Calculating SRMR", unit="file", leave=False):
        energy, r = process_file(recording)
        
        # Average along the time axis (axis 2) for each recording
        avg_energy = np.mean(energy, axis=2)
        
        # Sum up the average energy for each recording
        if total_energy is None:
            total_energy = avg_energy
        else:
            total_energy += avg_energy

    # Calculate the average energy across all recordings
    average_energy = total_energy / n_recordings
    return average_energy

BASE_PATH = '/home/sspowar/scratch/archive/LA/LA'

# TRAIN

train_df = pd.read_csv(f'{BASE_PATH}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
                       sep=" ", header=None)
train_df.columns =['speaker_id','filename','system_id','null','class_name']
train_df.drop(columns=['null'],inplace=True)
train_df['filepath'] = f'{BASE_PATH}/ASVspoof2019_LA_train/flac/'+train_df.filename+'.flac'
train_df['target'] = (train_df.class_name=='spoof').astype('int32') # set labels 1 for fake and 0 for real
# if DEBUG:
#     train_df = train_df.groupby(['target']).sample(2500).reset_index(drop=True)
print(f'Train Samples: {len(train_df)}')


# Separate real and fake recordings
real_recordings = train_df[train_df['target'] == 0]['filepath'].tolist()
fake_recordings = train_df[train_df['target'] == 1]['filepath'].tolist()

# Calculate average SRMR for real and fake recordings
average_real_energy = calculate_average_srmr(real_recordings)
print("Real Done")
average_fake_energy = calculate_average_srmr(fake_recordings)
print("Fake Done")

plot_srmr(average_real_energy, average_fake_energy, save_path='average_srmr_plot.png')

