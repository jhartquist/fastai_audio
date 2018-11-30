from pathlib import Path

from fastai.torch_core import to_np
import librosa
import numpy as np
import pytest
import torch

from fastai_audio.audio_clip import open_audio
from fastai_audio.transform import Spectrogram, ToDecibels, FrequencyToMel


DATA_PATH = Path('tests/data')


def get_data():
    # load data from example files
    clips = [open_audio(fn) for fn in DATA_PATH.iterdir()]
    sample_rate = clips[0].sample_rate
    tensors = [clip.data for clip in clips]
    # make them all the same length so they can be combined into a batch
    min_len = min(t.size(0) for t in tensors)
    tensors = [t[:min_len] for t in tensors]
    batch_tensor = torch.stack(tensors)
    return batch_tensor, sample_rate


def compute_np_batch(x_np, func, *args, **kwargs):
    return np.array([func(x, *args, **kwargs) for x in x_np])


def check_isclose(tensor, nparray, atol=1e-8):
    c = np.isclose(to_np(tensor), nparray, atol=atol)
    print(c.size, c.sum())
    assert np.isclose(to_np(tensor), nparray, atol=atol).all()


def stft(t, n_fft=1024, n_hop=256):
    return Spectrogram(n_fft=n_fft, n_hop=n_hop)(t)


def power_to_db(t, ref=1.0, top_db=None, amin=1e-7):
    return ToDecibels(ref=ref, top_db=top_db, amin=amin, normalized=False)(t)


def freq_to_mel(spec_t, n_mels=40, n_fft=1024, sr=16000, f_min=0.0, f_max=None):
    return FrequencyToMel(n_mels=n_mels, n_fft=n_fft, sr=sr,
                          f_min=f_min, f_max=f_max)(spec_t)


def ref_stft(x, n_fft, n_hop):
    spec = librosa.stft(x, n_fft=n_fft, hop_length=n_hop, center=True,
                        win_length=n_fft, window='hann', pad_mode='constant')
    mag, phase = librosa.magphase(spec)
    power = mag ** 2.0
    # torch.stft(normalized=True)
    norm_power = power / n_fft
    return norm_power


def ref_power_to_db(x, ref, top_db, amin):
    ref = np.max if ref == 'max' else ref
    return librosa.power_to_db(x, ref=ref, top_db=top_db, amin=amin)


def ref_freq_to_mel(spec_np, n_mels, sr, n_fft, f_min, f_max):
    return librosa.feature.melspectrogram(S=spec_np, sr=sr, n_fft=n_fft,
                                          n_mels=n_mels, power=2.0,
                                          fmin=f_min, fmax=f_max)


@pytest.mark.parametrize('n_fft', [512, 1024, 2048])
@pytest.mark.parametrize('n_hop', [128, 256, 512])
def test_stft(n_fft, n_hop):
    x_tensor, sr = get_data()
    x_np = to_np(x_tensor)

    stft_tensor = stft(x_tensor, n_fft=n_fft, n_hop=n_hop)
    stft_np = compute_np_batch(x_np, ref_stft, n_fft=n_fft, n_hop=n_hop)
    check_isclose(stft_tensor, stft_np)


@pytest.mark.parametrize('ref', [1.0, 'max'])
@pytest.mark.parametrize('top_db', [None, 40.0, 80.0, 100.0])
@pytest.mark.parametrize('amin', [1e-5, 1e-6, 1e-7])
def test_to_decibel(ref, top_db, amin):
    x_tensor, sr = get_data()
    stft_tensor = stft(x_tensor)
    stft_np = to_np(stft_tensor)

    db_tensor = power_to_db(stft_tensor,
                            ref=ref, top_db=top_db, amin=amin)
    db_np = compute_np_batch(stft_np, ref_power_to_db,
                             ref=ref, top_db=top_db, amin=amin)
    check_isclose(db_tensor, db_np, atol=amin*10)


@pytest.mark.parametrize('n_fft', [512, 1024, 2048])
@pytest.mark.parametrize('n_mels', [40, 64, 128])
@pytest.mark.parametrize('f_min', [0.0, 20.0])
@pytest.mark.parametrize('f_max', [None, 8000.0])
def test_freq_to_mel(n_fft, n_mels, f_min, f_max):
    x_tensor, sr = get_data()
    stft_tensor = stft(x_tensor, n_fft=n_fft)
    stft_np = to_np(stft_tensor)

    mel_tensor = freq_to_mel(stft_tensor, n_mels=n_mels, n_fft=n_fft, sr=sr,
                             f_min=f_min, f_max=f_max)
    mel_np = compute_np_batch(stft_np, ref_freq_to_mel, n_fft=n_fft,
                              n_mels=n_mels, sr=sr,
                              f_min=f_min, f_max=f_max)
    check_isclose(mel_tensor, mel_np)
