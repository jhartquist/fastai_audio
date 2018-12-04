import librosa as lr
from fastai import *

__all__ = ['get_frequency_transforms', 'get_frequency_batch_transforms',
           'FrequencyToMel', 'ToDecibels', 'Spectrogram', 'time_stretch']


TWO_PI = 2.0*np.pi


def get_frequency_transforms(n_fft=2048, n_hop=512, window=torch.hann_window,
                             n_mels=None, f_min=0, f_max=None, sample_rate=44100,
                             decibels=True, ref='max', top_db=80.0, norm_db=True,
                             rand_hop_pct=None):
    tfms = [Spectrogram(n_fft=n_fft, n_hop=n_hop, window=window, rand_hop_pct=rand_hop_pct)]
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
                 rand_hop_pct=None, device=None):
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.window = to_device(window(n_fft), device)
        if rand_hop_pct is not None:
            self.min_hop = int(self.n_hop * (1 - rand_hop_pct))
            self.max_hop = int(self.n_hop * (1 + rand_hop_pct))
        else:
            self.min_hop = self.max_hop = None

    def __call__(self, x):
        if self.min_hop is not None:
            n_hop = np.random.randint(self.min_hop, self.max_hop)
        else:
            n_hop = self.n_hop

        X = torch.stft(x,
                       n_fft=self.n_fft,
                       hop_length=n_hop,
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


def torch_stft(signal, n_fft=1024, hop_length=512, window=None, center=True, normalized=False):
    # pad signal
    if center:
        p = window.size(0) // 2
        signal = F.pad(signal, (p, p), 'constant')
    # overlap frames
    frames = signal.unfold(0, n_fft, hop_length) * window.unsqueeze(0)
    # compute ffts
    spectrum = torch.rfft(frames, 1)
    if normalized:
        spectrum.mul_(np.power(n_fft, -0.5))
    # tranpose to match torch.stft
    return spectrum


def torch_istft(spectrum, hop_length=512, window=None, center=True, length=None):
    n_fft, n_frames = spectrum.size(1), spectrum.size(0)
    n_fft = (n_fft - 1) * 2
    n_samples = (n_frames - 1) * hop_length + n_fft

    w = window.view(1, -1, 1)

    segments = torch.irfft(spectrum, 1, signal_sizes=(n_fft,))
    segments = segments.transpose(0,1)
    segments.unsqueeze_(0)
    segments.mul_(w)

    signal = F.fold(segments,
                    output_size=(1, n_samples),
                    kernel_size=(1, n_fft),
                    stride=(1, hop_length))

    norm = torch.ones_like(segments).mul_(w**2.0)
    norm = F.fold(norm,
                  output_size=(1, n_samples),
                  kernel_size=(1, n_fft),
                  stride=(1, hop_length))

    signal = signal.div_(norm).reshape(n_samples)

    # remove padding and trim to length
    start = 0
    end = n_samples
    if center:
        p = n_fft // 2
        start, end = p, -p
    if length is not None:
        end = start + length
    return signal[start:end]


def torch_phase_vocoder(mags, phases, rate, hop_length=None):
    n_steps, n_fft_half = mags.size()
    n_fft = 2 * (n_fft_half - 1)

    if hop_length is None:
        hop_length = int(n_fft // 4)

    # TODO: modified this from n_steps due to index out of bounds on line 183
    time_steps = torch.arange(0, n_steps-1, rate)
    time_steps_idx = time_steps.long()
    alpha = torch.remainder(time_steps, 1.0).unsqueeze_(1)

    mags = F.pad(mags, [0,0,0,1])
    new_mags = ((1.0 - alpha) * mags[time_steps_idx, :]
                + alpha  * mags[time_steps_idx+1, :])


    initial_phase = phases[:1,:]
    phase_delta = phases[1:,:] - phases[:-1,:]

    phi_advance = torch.linspace(0, np.pi * hop_length, n_fft_half).unsqueeze_(0)
    phase_delta.sub_(phi_advance)

    # wrap to [-pi, pi] range
    phase_delta.sub_(TWO_PI * torch.round(phase_delta / TWO_PI))

    new_phase_delta = phase_delta[time_steps_idx[:-1]]
    new_phase_delta.add_(phi_advance)
    new_phases = torch.cumsum(torch.cat([initial_phase,
                                         new_phase_delta], dim=0), dim=0)
    return new_mags, new_phases


def time_stretch(x, rate, n_fft=2048, n_hop=512):
    w = torch.hann_window(n_fft)

    X = torch_stft(x, n_fft=n_fft, hop_length=n_hop, window=w)

    # rect -> polar
    X_squared = X.pow(2.0)
    Xm = (X_squared[...,0] + X_squared[...,1]).sqrt_()
    Xp = X[...,1].atan2_(X[...,0])

    # actual stretching
    Xmh, Xph = torch_phase_vocoder(Xm, Xp, rate, hop_length=n_hop)

    # polar -> rect
    Xh = torch.empty(*Xmh.size(), 2)
    torch.mul(Xmh, Xph.cos(), out=Xh[:,:,0])
    torch.mul(Xmh, Xph.sin(), out=Xh[:,:,1])

    xh = torch_istft(Xh, hop_length=n_hop, window=w)
    return xh
