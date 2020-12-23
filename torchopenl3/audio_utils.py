import numpy as np
import librosa
import yaml
import warnings
import torch

def _check_audio_types(audio: np.ndarray):
    assert isinstance(audio, np.ndarray), f'expected np.ndarray but got {type(audio)} as input.'
    assert audio.ndim == 2, f'audio must be shape (channels, time), got shape {audio.shape}'
    if audio.shape[-1] < audio.shape[-2]:
        warnings.warn(f'got audio shape {audio.shape}. Audio should be (channels, time). \
                        typically, the number of samples is much larger than the number of channels. ')
    if _is_zero(audio):
        warnings.warn(f'provided audio array is all zeros')

def _is_mono(audio: np.ndarray):
    _check_audio_types(audio)
    num_channels = audio.shape[-2]
    return num_channels == 1

def _is_zero(audio: np.ndarray):
    return np.all(audio == 0);

def load_audio_file(path_to_audio, sample_rate=48000):
    """ wrapper for loading monophonic audio with librosa
    Args:
        path_to_audio (str): path to audio file
        sample_rate (int): target sample rate
    returns:
        audio (np.ndarray): monophonic audio with shape (samples,)
    """
    audio, sr = librosa.load(path_to_audio, mono=True, sr=sample_rate)
    # add channel dimension
    audio = np.expand_dims(audio, axis=-2)
    return audio

def window(audio: np.ndarray, window_len: int = 48000, hop_len: int = 4800):
    """split audio into overlapping windows

    note: this is not a memory efficient view like librosa.util.frame. 
    It will return a new copy of the array

    Args:
        audio (np.ndarray): audio array with shape (channels, samples)
        window_len (int, optional): [description]. Defaults to 48000.
        hop_len (int, optional): [description]. Defaults to 4800.
    Returns:
        audio_windowed (np.ndarray): windowed audio array with shape (frame, channels, samples)
    """
    _check_audio_types(audio)
    # determine how many window_len windows we can get out of the audio array
    # use ceil because we can zero pad
    n_chunks = int(np.ceil(len(audio)/(window_len))) 
    start_idxs = np.arange(0, n_chunks * window_len, hop_len)

    windows = []
    for start_idx in start_idxs:
        # end index should be start index + window length
        end_idx = start_idx + window_len
        # BUT, if we have reached the end of the audio, stop there
        end_idx = min([end_idx, len(audio)])
        # create audio window
        win = np.array(audio[:, start_idx:end_idx])
        # zero pad window if needed
        win = zero_pad(win, required_len=window_len)
        windows.append(win)
    
    audio_windowed = np.stack(windows)
    return audio_windowed

def downmix(audio: np.ndarray):
    """ downmix an audio array.
    must be shape (channels, mono)

    Args:
        audio ([np.ndarray]): array to downmix
    """
    _check_audio_types(audio)
    audio = audio.mean(axis=-2, keepdims=True)
    return audio

def resample(audio: np.ndarray, old_sr: int, new_sr: int = 48000) -> np.ndarray:
    """wrapper around librosa for resampling

    Args:
        audio (np.ndarray): audio array shape (channels, time)
        old_sr (int): old sample rate
        new_sr (int, optional): target sample rate.  Defaults to 48000.

    Returns:
        np.darray: resampled audio. shape (channels, time)
    """
    _check_audio_types(audio)

    if _is_mono(audio):
        audio = audio[0]
        audio = librosa.resample(audio, old_sr, new_sr)
        audio = np.expand_dims(audio, axis=-2)
    else:
        audio = librosa.resample(audio, old_sr, new_sr)
    return audio

def zero_pad(audio: np.ndarray, required_len: int = 48000) -> np.ndarray:
    """zero pad audio array to meet a multiple of required_len

    Args:
        audio (np.ndarray): audio array w shape (channels, sample)
        required_len (int, optional): target length in samples. Defaults to 48000.

    Returns:
        np.ndarray: zero padded audio
    """
    _check_audio_types(audio)

    num_frames = audio.shape[-1]

    before = 0
    after = required_len - num_frames%required_len
    if after == required_len:
        return audio
    audio = np.pad(audio, pad_width=((0, 0), (before, after)), mode='constant', constant_values=0)
    return audio