from fastai.torch_core import *

__all__ = ['mapk']


def mapk_np(preds, targs, k=3):
    preds = np.argsort(-preds, axis=1)[:, :k]
    score = 0.
    for i in range(k):
        num_hits = (preds[:, i] == targs).sum()
        score += num_hits * (1. / (i+1.))
    score /= preds.shape[0]
    return score


def mapk(preds, targs, k=3):
    return tensor(mapk_np(to_np(preds), to_np(targs), k))
