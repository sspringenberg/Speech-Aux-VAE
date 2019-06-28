import ujson
from tqdm import tqdm
import torch.utils.data as data
import torch
from torch.nn import functional as F
import torchaudio
from torchaudio import datasets, transforms, save
import librosa
import numpy as np
import sys

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import pylab

import librosa.display

def read_audio(fp, downsample=True):
    sig, sr = librosa.load(fp)
    if downsample:
        # 22khz -> 11 khz
        if sig.size % 2 == 0:
            sig = sig[::2]
            sr = sr/2
        else:
            sig = sig[:-(sig.size % 2):2]
            sr = sr/2
    
    sig = torch.FloatTensor(sig)
    return sig, sr

def pad_trim(sig, max_len):
    if sig.size(0) > max_len:
        sig = sig[:max_len]
    elif sig.size(0) < max_len:
        sig = F.pad(sig, (0,max_len-sig.size(0)), "constant", 0)
    return sig
        
def stft(y, sr, n_fft=400, hop_t=0.010, win_t=0.025, window="hamming",
        preemphasis=0.97):
    """
    Short time Fourier Transform
    Args:
        y(np.ndarray): raw waveform of shape (T,)
        sr(int): sample rate
        hop_t(float): spacing (in second) between consecutive frames
        win_t(float): window size (in second)
        window(str): type of window applied for STFT
        preemphasis(float): pre-emphasize raw signal with y[t] = x[t] - r*x[t-1]
    Return:
        (np.ndarray): (n_fft / 2 + 1, N) matrix; N is number of frames
    """
    if preemphasis > 1e-12:
        y = y - preemphasis * np.concatenate([[0], y[:-1]], 0)
    hop_length = int(sr * hop_t)
    win_length = int(sr * win_t)
    spec = np.abs(librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window))
    thresh=0.01
    spec[spec < thresh] = thresh
    return spec
  
def to_audio(y, sr, n_fft=1024, hop_t=0.010, win_t=0.025):
    
    hop_length = int(sr * hop_t)
    win_length = int(sr * win_t)
    
    spec = griffinlim(librosa.db_to_amplitude(y), n_iter = 100, win_length = win_length, n_fft = n_fft, hop_length = hop_length)
    
    return spec

def to_melspec(y, sr, n_fft=400, hop_t=0.010, win_t=0.025, window="hamming", 
        preemphasis=0.97, n_mels=80, log=False, norm_mel=1, log_floor=-20):
    """
    Compute Mel-scale filter bank coefficients:
    Args:
        y(np.ndarray): 
        sr(int):
        hop_t(float):
        win_t(float):
        window(str):
        preemphasis(float):
        n_mels(int): number of filter banks, which are equally spaced in Mel-scale
        log(bool):
        norm_mel(None/1): normalize each filter bank to have area of 1 if set to 1;
            otherwise the peak value of eahc filter bank is 1
    Return:
        (np.ndarray): (n_mels, N) matrix; N is number of frames
    """
    hop_length = int(sr * hop_t)
    spec = stft(y, sr, n_fft, hop_t, win_t, window, preemphasis, log=False)
    melspec = librosa.feature.melspectrogram(sr=sr, S=spec, n_fft=n_fft, 
                                             hop_length=hop_length, n_mels=n_mels, norm=norm_mel, fmax=8000)
    melspec = melspec/np.max(melspec)
    return melspec

def to_spec(y, sr, n_fft=400, hop_t=0.010, win_t=0.025, window="hamming", preemphasis=0.97):
    hop_len = int(hop_t*sr)
    win_len = int(win_t*sr)
    
    spec = stft(y, sr, n_fft=n_fft, hop_t=hop_t, win_t=win_t, preemphasis=preemphasis)
    spec = librosa.amplitude_to_db(spec, ref=np.max)
    return spec

def energy_vad(y, sr, hop_t=0.010, win_t=0.025, th_ratio=1.04/2):
    """
    Compute energy-based VAD
    """
    hop_length = int(sr * hop_t)
    win_length = int(sr * win_t)
    e = librosa.feature.rmse(y, frame_length=win_length, hop_length=hop_length)
    th = th_ratio * np.mean(e)
    vad = np.asarray(e > th, dtype=int)
    return vad

def griffinlim(spectrogram, n_fft, win_length, n_iter = 100, window="hamming", hop_length = -1, verbose = False):
    if hop_length == -1:
        hop_length = n_fft // 4
    
    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))
    
    t = tqdm(range(n_iter), ncols=100, mininterval=2.0, disable=not verbose)
    for i in t:
        full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = librosa.istft(full, hop_length = hop_length, window = window, win_length = win_length)
        rebuilt = librosa.stft(inverse, n_fft = n_fft, hop_length = hop_length, window = window, win_length = win_length)
        angles = np.exp(1j * np.angle(rebuilt))

        if verbose:
            diff = np.abs(spectrogram) - np.abs(rebuilt)
            t.set_postfix(loss=np.linalg.norm(diff, 'fro'))

    full = np.abs(spectrogram).astype(np.complex) * angles
    inverse = librosa.istft(full, hop_length = hop_length, window = window, win_length = win_length)

    return inverse


class DB_MelSpec(object):
    
    def __init__(self, sr=16000, ffts=1024, n_fft=1024, hop_t=0.010, win_t=0.025, n_mels=80, preemphasis=0.97, **kwargs):        
        self.sr = sr
        self.n_fft = n_fft
        self.hop_t = hop_t
        self.win_t = win_t
        self.n_mels = n_mels
        self.preemphasis = preemphasis
        self.kwargs = kwargs

    def __call__(self, tensor):
        
        sr = self.sr
        n_fft = self.n_fft
        hop_t = self.hop_t
        win_t = self.win_t
        preemphasis=self.preemphasis
        
        tensor = tensor.view(-1)
        tensor = tensor.numpy()
        
        tensor = to_melspec(tensor, sr=sr, n_fft=n_fft, hop_t=hop_t, win_t=win_t, log=False, preemphasis=preemphasis)
        tensor = librosa.power_to_db(tensor, ref=np.max)
        tensor = torch.from_numpy(tensor).float()
        
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '()'

class DB_Spec(object):
    
    def __init__(self, sr=16000, n_fft=1024, hop_t=0.010, win_t=0.025, preemphasis=0.2, max_len=None, **kwargs):  
        self.sr = sr
        self.n_fft = n_fft
        self.hop_t = hop_t
        self.win_t = win_t
        self.preemphasis = preemphasis
        self.kwargs = kwargs
        self.max_len = max_len

    def __call__(self, tensor):
        
        sr = self.sr
        n_fft = self.n_fft
        hop_t = self.hop_t
        win_t = self.win_t
        preemphasis = self.preemphasis
        
        if self.max_len is not None:
            tensor = pad_trim(tensor, max_len=self.max_len)
        tensor = tensor.numpy()
        tensor = to_spec(tensor, sr=sr, n_fft=n_fft, hop_t=hop_t, win_t=win_t, preemphasis=preemphasis)
        
        tensor = torch.from_numpy(tensor).float()
        
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
