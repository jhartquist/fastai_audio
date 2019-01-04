import librosa as lr
from fastai.torch_core import *

__all__ = ['get_frequency_transforms', 'get_frequency_batch_transforms',
           'FrequencyToMel', 'ToDecibels', 'Spectrogram']


def get_frequency_transforms(n_fft=2048, n_hop=512, window=torch.hann_window,
                             n_mels=None, f_min=0, f_max=None, sample_rate=44100,
                             decibels=True, ref='max', top_db=80.0, norm_db=True):
    tfms = [Spectrogram(n_fft=n_fft, n_hop=n_hop, window=window)]
    if n_mels is not None:
        tfms.append(FrequencyToMel(n_mels=n_mels, n_fft=n_fft, sr=sample_rate,
                                   f_min=f_min, f_max=f_max))
    if decibels:
        tfms.append(ToDecibels(ref=ref, top_db=top_db, normalized=norm_db))

    # only one list, as its applied to all dataloaders
    return tfms


def get_frequency_batch_transforms(*args, add_channel_dim=True, **kwargs):
    tfms = get_frequency_transforms(*args, **kwargs)

    def _freq_batch_transformer(inputs):
        xs, ys = inputs
        for tfm in tfms:
            xs = tfm(xs)
        if add_channel_dim:
            xs.unsqueeze_(1)
        return xs, ys
    return [_freq_batch_transformer]


class FrequencyToMel:
    def __init__(self, n_mels=40, n_fft=1024, sr=16000,
                 f_min=0.0, f_max=None, device=None):
        mel_fb = lr.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                fmin=f_min, fmax=f_max).astype(np.float32)
        self.mel_filterbank = to_device(torch.from_numpy(mel_fb), device)

    def __call__(self, spec_f):
        spec_m = self.mel_filterbank @ spec_f
        return spec_m


class ToDecibels:
    def __init__(self,
                 power=2, # magnitude=1, power=2
                 ref=1.0,
                 top_db=None,
                 normalized=True,
                 amin=1e-7):
        self.constant = 10.0 if power == 2 else 20.0
        self.ref = ref
        self.top_db = abs(top_db) if top_db else top_db
        self.normalized = normalized
        self.amin = amin

    def __call__(self, x):
        batch_size = x.shape[0]
        if self.ref == 'max':
            ref_value = x.contiguous().view(batch_size, -1).max(dim=-1)[0]
            ref_value.unsqueeze_(1).unsqueeze_(1)
        else:
            ref_value = tensor(self.ref)
        spec_db = x.clamp_min(self.amin).log10_().mul_(self.constant)
        spec_db.sub_(ref_value.clamp_min_(self.amin).log10_().mul_(10.0))
        if self.top_db is not None:
            max_spec = spec_db.view(batch_size, -1).max(dim=-1)[0]
            max_spec.unsqueeze_(1).unsqueeze_(1)
            spec_db = torch.max(spec_db, max_spec - self.top_db)
            if self.normalized:
                # normalize to [0, 1]
                spec_db.add_(self.top_db).div_(self.top_db)
        return spec_db


# Returns power spectrogram (magnitude squared)
class Spectrogram:
    def __init__(self, n_fft=1024, n_hop=256, window=torch.hann_window,
                 device=None):
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.window = to_device(window(n_fft), device)

    def __call__(self, x):
        X = torch.stft(x,
                       n_fft=self.n_fft,
                       hop_length=self.n_hop,
                       win_length=self.n_fft,
                       window=self.window,
                       onesided=True,
                       center=True,
                       pad_mode='constant',
                       normalized=True)
        # compute power from real and imag parts (magnitude^2)
        X.pow_(2.0)
        power = X[:,:,:,0] + X[:,:,:,1]
        return power
