# data_processing.py
import numpy as np
from scipy.signal import hamming
from srmrpy.hilbert import hilbert
from srmrpy.modulation_filters import *
from gammatone.fftweight import fft_gtgram
from gammatone.filters import centre_freqs, make_erb_filters, erb_filterbank
from srmrpy.segmentaxis import segment_axis
import soundfile as sf

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

# EER Calculation 
def calculate_eer(y_true, y_scores):
    fpr, fnr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - fnr)))]
    return eer