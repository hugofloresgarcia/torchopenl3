import os

import numpy as np
import torch
import torchopenl3
from torchopenl3 import OpenL3Embedding

import torchopenl3.audio_utils as audio_utils

def embed(model: OpenL3Embedding, audio: np.ndarray, sample_rate: int, hop_size: int = 1):
    """compute OpenL3 embeddings for a given audio array. 
    Args:
        model (OpenL3Embedding): OpenL3 model to use
        audio (np.ndarray): audio array with shape (channels, samples). 
                            audio will be downmixed to mono. 
        sample_rate (int): input sample rate. Will be resampled to torchopenl3.SAMPLE_RATE
        hop_size (int, optional): hop size, in seconds. Defaults to 1.
    Returns:
        np.ndarray: embeddings with shape (frame, features)
    """
    audio_utils._check_audio_types(audio)
    # resample, downmix, and zero pad if needed
    audio = audio_utils.resample(audio, sample_rate, torchopenl3.SAMPLE_RATE)
    audio = audio_utils.downmix(audio)

    # split audio into overlapping windows as dictated by hop_size
    hop_len: int = hop_size * torchopenl3.SAMPLE_RATE
    audio = audio_utils.window(audio, window_len=1*torchopenl3.SAMPLE_RATE, hop_len=hop_len)

    # convert to torch tensor!
    audio = torch.from_numpy(audio)

    # TODO: add GPU support
    model.eval()
    with torch.no_grad():
        embeddings = model(audio)
    
    return embeddings.cpu().numpy()

def embed_from_file_to_array(model: OpenL3Embedding, path_to_audio: str, hop_size: int = 1):
    """compute OpenL3 embeddings from a given audio file

    Args:
        model (OpenL3Embedding): model to embed with
        path_to_audio (str): path to audio file
        hop_size (int, optional): embedding hop size, in seconds. Defaults to 1.

    Returns:
        np.ndarray: embeddings with shape (frame, features)
    """
    # load audio
    audio = audio_utils.load_audio_file(path_to_audio, sample_rate=torchopenl3.SAMPLE_RATE)
    return embed(model, audio, torchopenl3.SAMPLE_RATE, hop_size)

def embed_from_file_to_file(model: OpenL3Embedding, path_to_audio: str, path_to_output: str, hop_size: int = 1):
    """compute OpenL3 embeddings from a given audio file and save to an output path in .npy format

    Args:
        model (OpenL3Embedding): model to embed with
        path_to_audio (str): path to audio file
        path_to_output (str): path to output (.npy) file
        hop_size (int, optional): embedding hop size, in seconds. Defaults to 1.
    """
    # get embedding array
    embeddings = embed_from_file_to_array(model, path_to_audio, hop_size)
    np.save(path_to_output, embeddings)