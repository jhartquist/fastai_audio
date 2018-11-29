from fastai import *
from fastai.text import SortSampler, SortishSampler
from .audio_clip import *

__all__ = ['pad_collate', 'AudioDataBunch', 'AudioItemList', ]


def pad_collate(batch):
    xs, ys = zip(*to_data(batch))
    max_len = max(x.size(0) for x in xs)
    padded_xs = torch.zeros(len(xs), max_len, dtype=xs[0].dtype)
    for i,x in enumerate(xs):
        padded_xs[i,:x.size(0)] = x
    return padded_xs, tensor(ys)


class AudioDataBunch(DataBunch):
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path='.',
               bs=64, equal_lengths=True, tfms=None, **kwargs):
        if equal_lengths:
            return super().create(train_ds, valid_ds, test_ds=test_ds, path=path,
                                  bs=bs, tfms=tfms, **kwargs)
        else:
            datasets = [train_ds, valid_ds]
            if test_ds is not None:
                datasets.append(test_ds)

            train_sampler = SortishSampler(datasets[0].x,
                                           key=lambda i: datasets[0][i][0].data.shape[0],
                                           bs=bs)
            train_dl = DataLoader(datasets[0], batch_size=bs, sampler=train_sampler, **kwargs)
            dataloaders = [train_dl]
            for ds in datasets[1:]:
                sampler = SortSampler(ds.x,
                                      key=lambda i: ds[i][0].data.shape[0])
                dataloaders.append(DataLoader(ds, batch_size=bs, sampler=sampler, **kwargs))
            return cls(*dataloaders, path=path, collate_fn=pad_collate, tfms=tfms)

    def show_batch(self, rows:int=5, ds_type:DatasetType=DatasetType.Train, **kwargs):
        dl = self.dl(ds_type)
        ds = dl.dl.dataset
        idx = np.random.choice(len(ds), size=rows, replace=False)
        batch = ds[idx]
        xs, ys = batch.x, batch.y
        self.train_ds.show_xys(xs, ys, **kwargs)


class AudioItemList(ItemList):
    """NOTE: this class has been heavily adapted from ImageItemList"""
    _bunch = AudioDataBunch

    @classmethod
    def open(cls, fn):
        return open_audio(fn)

    def get(self, i):
        fn = super().get(i)
        return self.open(fn)

    @classmethod
    def from_df(cls, df, path, col=0, folder='.', suffix=''):
        """Get the filenames in `col` of `df` and will had `path/folder` in front of them,
        `suffix` at the end. `create_func` is used to open the audio files."""
        suffix = suffix or ''
        res = super().from_df(df, path=path, col=col)
        res.items = np.char.add(np.char.add(f'{folder}/', res.items.astype(str)), suffix)
        res.items = np.char.add(f'{res.path}/', res.items)
        return res

    def show_xys(self, xs, ys, figsize=None, **kwargs):
        for x, y in zip(xs, ys):
            x.show(title=y, **kwargs)

