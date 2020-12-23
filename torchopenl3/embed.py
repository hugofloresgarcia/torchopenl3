import os

import numpy as np
import torch
import torchopenl3
from torchopenl3 import OpenL3Embedding

import .audio_utils as audio_utils

def embed(model: OpenL3Embedding, audio: np.ndarray, sample_rate: int, hop_size: int = 1):
    """compute OpenL3 embeddings for a given audio array. 
    Args:
        model (OpenL3Embedding): OpenL3 model to use
        audio (np.ndarray): audio array with shape (channels, samples). 
                            audio will be downmixed to mono. 
        sample_rate (int): input sample rate. Will be resampled to torchopenl3.SAMPLE_RATE
        hop_size (int, optional): hop size, in seconds. Defaults to 1.
    Returns:
        np.ndarray: [description]
    """
    audio_utils._check_audio_types(audio)
    # resample, downmix, and zero pad if needed
    audio = audio_utils.resample(audio, sample_rate, torchopenl3.SAMPLE_RATE)
    audio = audio_utils.downmix(audio)

    # split audio into overlapping windows as dictated by hop_size
    hop_len: int = hop_size * torchopenl3.SAMPLE_RATE
    audio = audio_utils.window(audio, window_len=1*torchopenl3.SAMPLE_RATE, hop_len)

    # convert to torch tensor!
    audio = torch.from_numpy(audio)

    with torch.no_grad():
        embeddings = model(audio)
    
    return embeddings


