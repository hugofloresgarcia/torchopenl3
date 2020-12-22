import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt

def get_stft_filterbank(N, window='hann'):
    """
    returns real and imaginary kernels for a dft filterbank
    an assymetric window is used (see librosa.filters.get_window)
    params
    ------
    N (int): number of dft components
    returns
    -------
    f_real (np.ndarray): real filterbank with shape (N, N//2+1)
    f_imag (np.ndarray): imag filterbank with shape (N, N//2+1)
    """
    K = N // 2 + 1 
    # discrete time axis 
    n = np.arange(N)

    w_k = np.arange(K) * 2  * np.pi / float(N)
    f_real = np.cos(w_k.reshape(-1, 1) * n.reshape(1, -1))
    f_imag = np.sin(w_k.reshape(-1, 1) * n.reshape(1, -1))

    window = librosa.filters.get_window(window, N, fftbins=True)
    window = window.reshape((1, -1))

    f_real = np.multiply(f_real, window)
    f_imag = np.multiply(f_imag, window)

    return f_real, f_imag

def amplitude_to_db(
        x: torch.Tensor, amin: float = 1e-10, 
        dynamic_range: float = 80.0, 
        to_torchscript: bool = True):
    """
    per kapre's amplitude to db
    for torchscript compiling reasons
        x must be shape (batch, channels, height, width)
    update: use to_torchscript flag as false to use different array shapes
    """
    
    # apply log transformation (amplitude to db)
    amin = torch.full_like(x, 1e-10).float()
    log10 = torch.tensor(np.log(10)).float()
    x = 10 * torch.log(torch.max(x, amin)) / log10
        
    xmax, v = x.max(dim=1, keepdim=True)
    xmax, v = xmax.max(dim=2, keepdim=True)
    xmax, v = xmax.max(dim=3, keepdim=True)

    x = x - xmax 

    x = x.clamp(min=float(-dynamic_range), max=None) # [-80, 0]
    return x

class Melspectrogram(pl.LightningModule):

    def __init__(self, 
                sr=48000, n_mels=128, fmin=0.0, fmax=None, 
                power_melgram=1.0, return_decibel_melgram=True, 
                trainable_fb=False, htk=True):
        """
        creates a single 1D convolutional layers with filters fixed to
        a mel filterbank.
        """
        #TODO: make two separate classes for spec and melspec ala kapre
        super().__init__()
        if fmax is None:
            fmax = sr / 2
        
        self.sr = sr
        # scale some parameters according to openl3
        self.sr_scale = self.sr // 48000
        self.n_fft = int(self.sr_scale  * 2048)
        self.hop_size = int(self.sr_scale * 242)
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.htk = htk
        self.return_decibel_melgram = return_decibel_melgram
        self.power_melgram = power_melgram
        

        f_real, f_imag = get_stft_filterbank(self.n_fft, window='hann')
        self.n_bins = self.n_fft // 2 + 1
        
        mel_filters = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=n_mels,
                                        fmin=self.fmin, fmax=self.fmax, htk=self.htk) # (mel, 1+n_fft/2)
        
        self.conv1d_real = nn.Conv1d(
            in_channels=1, 
            out_channels=self.n_bins,
            kernel_size=self.n_fft,
            stride=self.hop_size, 
            padding=1101,
            bias=False)

        self.conv1d_imag = nn.Conv1d(
            in_channels=1, 
            out_channels=self.n_bins,
            kernel_size=self.n_fft,
            stride=self.hop_size, 
            padding=1101,
            bias=False)

        self.freq2mel = nn.Linear(self.n_bins, self.n_mels, bias=False)

        # fix the weights to value
        f_real = torch.from_numpy(f_real).float()
        f_imag = torch.from_numpy(f_imag).float()
        mel_filters = torch.from_numpy(mel_filters).float()

        self.mel_filters = nn.Parameter(mel_filters)
        self.conv1d_real.weight = nn.Parameter(f_real.unsqueeze(1))
        self.conv1d_imag.weight = nn.Parameter(f_imag.unsqueeze(1))
        self.freq2mel.weights = nn.Parameter(mel_filters)

    def forward(self, x):
        if not self.trainable_fb:
            self.freeze()

        # input should be shape (batch, channels, time)
        # STFT and mel filters
        # forward pass through filterbank
        real = self.conv1d_real(x)
        imag = self.conv1d_imag(x)

        x = real ** 2 + imag ** 2
        
        x = x.permute(0, 2, 1)
        x = torch.matmul(x, self.mel_filters.T)
        x = x.permute(0, 2, 1)

        # NOW, take the square root to make it a power 1 melgram
        x = torch.pow(torch.sqrt(x), self.power_melgram)

        x = x.view(-1, 1, 128, 199)
        
        if self.return_decibel_melgram:
            x = amplitude_to_db(x)
        return x