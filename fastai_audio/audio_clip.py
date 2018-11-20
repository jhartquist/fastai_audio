from fastai import *
from scipy.io import wavfile

__all__ = ['AudioClip', 'open_audio']


class AudioClip(ItemBase):
    def __init__(self, samples, sample_rate):
        self.data = samples
        self.sample_rate = sample_rate

    def apply_tfms(self, tfms, **kwargs):
        if tfms:
            print('apply_tfms')
        return self


def open_audio(fn):
    sr, x = wavfile.read(fn)
    t = torch.from_numpy(x.astype(np.float32, copy=False))
    if x.dtype == np.int16:
        t.div_(32767)
    elif x.dtype != np.float32:
        raise OSError('Encountered unexpected dtype: {}'.format(x.dtype))
    return AudioClip(t, sr)
