from fastai import *
from scipy.io import wavfile

__all__ = ['AudioClip', 'open_audio']


class AudioClip(ItemBase):
    def __init__(self, signal, sample_rate):
        self.data = signal
        self.sample_rate = sample_rate

    def __str__(self):
        return '(duration={}s, sample_rate={:.1f}KHz)'.format(
            self.duration, self.sample_rate/1000)

    @property
    def num_samples(self):
        return len(self.data)

    @property
    def duration(self):
        return self.num_samples / self.sample_rate


def open_audio(fn):
    sr, x = wavfile.read(fn)
    t = torch.from_numpy(x.astype(np.float32, copy=False))
    if x.dtype == np.int16:
        t.div_(32767)
    elif x.dtype != np.float32:
        raise OSError('Encountered unexpected dtype: {}'.format(x.dtype))
    return AudioClip(t, sr)
