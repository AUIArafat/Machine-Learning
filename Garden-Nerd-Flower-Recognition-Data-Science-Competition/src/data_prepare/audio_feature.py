import librosa
import numpy as np


def load_audio(path, sr=44100, offset=0, duration=3):
    """
    read audio file at given sampling rate
    sr: target sampling rate, default sampling rate is 44.1 kHz, as in paper
    https://arxiv.org/abs/1608.04363
    offset: start reading after this time (in seconds)
    duration: only load up to this much audio (in seconds)
    """
    y, _ = librosa.load(path, sr=sr, offset=offset, duration=duration)
    return y


def get_mel_spectrogram(y, sr=44100, n_fft=1024, hop_length=1024, n_mels=128, fmin=0, fmax=22050, use_db=True):
    """
    compute a mel-scaled spectrogram
    default values for sr, n_fft, hop_length, n_mels, fmin and fmax are as in the paper https://arxiv.org/abs/1608.04363
    if use_db is True use log scaled mel spectrogram
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin,
                                         fmax=fmax)
    if use_db:
        mel = librosa.power_to_db(mel, ref=np.max)
    return mel




