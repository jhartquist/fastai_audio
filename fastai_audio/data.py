from fastai.basic_data import *
from fastai.data_block import *
from fastai.data_block import _maybe_squeeze
from fastai.text import SortSampler, SortishSampler
from fastai.torch_core import *
from .audio_clip import *

__all__ = ['pad_collate1d', 'pad_collate2d', 'AudioDataBunch', 'AudioItemList', ]


def pad_collate1d(batch):
    xs, ys = zip(*to_data(batch))
    max_len = max(x.size(0) for x in xs)
    padded_xs = torch.zeros(len(xs), max_len, dtype=xs[0].dtype)
    for i,x in enumerate(xs):
        padded_xs[i,:x.size(0)] = x
    return padded_xs, tensor(ys)


# TODO: generalize this away from hard coding dim values
def pad_collate2d(batch):
    xs, ys = zip(*to_data(batch))
    max_len = max(max(x.size(1) for x in xs), 1)
    bins = xs[0].size(0)
    padded_xs = torch.zeros(len(xs), bins, max_len, dtype=xs[0].dtype)
    for i,x in enumerate(xs):
        padded_xs[i,:,:x.size(1)] = x
    return padded_xs, tensor(ys)


class AudioDataBunch(DataBunch):
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path='.',
               bs=64, equal_lengths=True, length_col=None, tfms=None, **kwargs):
        if equal_lengths:
            return super().create(train_ds, valid_ds, test_ds=test_ds, path=path,
                                  bs=bs, tfms=tfms, **kwargs)
        else:
            datasets = super()._init_ds(train_ds, valid_ds, test_ds)
            train_ds, valid_ds, fix_ds = datasets[:3]
            if len(datasets) == 4:
                test_ds = datasets[3]

            train_lengths = train_ds.lengths(length_col)
            train_sampler = SortishSampler(train_ds.x, key=lambda i: train_lengths[i], bs=bs//2)
            train_dl = DataLoader(train_ds, batch_size=bs, sampler=train_sampler, **kwargs)

            # precalculate lengths ahead of time if they aren't included in xtra
            valid_lengths = valid_ds.lengths(length_col)
            valid_sampler = SortSampler(valid_ds.x, key=lambda i: valid_lengths[i])
            valid_dl = DataLoader(valid_ds, batch_size=bs, sampler=valid_sampler, **kwargs)

            fix_lengths = fix_ds.lengths(length_col)
            fix_sampler = SortSampler(fix_ds.x, key=lambda i: fix_lengths[i])
            fix_dl = DataLoader(fix_ds, batch_size=bs, sampler=fix_sampler, **kwargs)

            dataloaders = [train_dl, valid_dl, fix_dl]
            if test_ds is not None:
                test_dl = DataLoader(test_ds, batch_size=1, **kwargs)
                dataloaders.append(test_dl)

            return cls(*dataloaders, path=path, collate_fn=pad_collate1d, tfms=tfms)

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

    def lengths(self, length_col=None):
        if length_col is not None and self.xtra is not None:
            lengths = self.xtra.iloc[:, df_names_to_idx(length_col, self.xtra)]
            lengths = _maybe_squeeze(lengths.values)
        else:
            lengths = [clip.num_samples for clip in self]
        return lengths

    @classmethod
    def from_df(cls, df, path, col=0, folder='.', suffix='', length_col=None):
        """Get the filenames in `col` of `df` and will had `path/folder` in front of them,
        `suffix` at the end. `create_func` is used to open the audio files."""
        suffix = suffix or ''
        res = super().from_df(df, path=path, col=col, length_col=length_col)
        res.items = np.char.add(np.char.add(f'{folder}/', res.items.astype(str)), suffix)
        res.items = np.char.add(f'{res.path}/', res.items)
        return res

    def show_xys(self, xs, ys, figsize=None, **kwargs):
        for x, y in zip(xs, ys):
            x.show(title=y, **kwargs)

