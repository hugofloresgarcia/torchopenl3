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
        hop_size (int, optional): [description]. Defaults to 1.
    Returns:
        np.ndarray: [description]
    """
    audio_utils._check_audio_types(audio)
    # resample, downmix, and zero pad if needed
    audio = audio_utils.resample(audio, sample_rate, torchopenl3.SAMPLE_RATE)
    audio = audio_utils.downmix(audio)
    audio = audio_utils.zero_pad(audio)

    # convert to torch tensor!
    audio = torch.from_numpy(audio)

    # TODO: need to enforce a maximum batch size
    # to avoid OOM errors
    audio = audio.view(-1, 1, torchopenl3.sample_rate)

    with torch.no_grad():
        embeddings = model(audio)
    
    return embeddings


