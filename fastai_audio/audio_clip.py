from fastai import *
from scipy.io import wavfile
from IPython.display import display, Audio

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

    def show(self, ax=None, figsize=(5, 1), player=True, title=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        if title:
            ax.set_title(title)
        timesteps = np.arange(len(self.data)) / self.sample_rate
        ax.plot(timesteps, self.data)
        ax.set_xlabel('Time (s)')
        plt.show()
        if player:
            # unable to display an IPython 'Audio' player in plt axes
            display(Audio(self.data, rate=self.sample_rate))

    def show_batch(self, idxs, rows, ds, **kwargs):
        for i in idxs[:rows]:
            x,y = ds[i]
            x.show(title=y, **kwargs)


def open_audio(fn):
    sr, x = wavfile.read(fn)
    t = torch.from_numpy(x.astype(np.float32, copy=False))
    if x.dtype == np.int16:
        t.div_(32767)
    elif x.dtype != np.float32:
        raise OSError('Encountered unexpected dtype: {}'.format(x.dtype))
    return AudioClip(t, sr)
