"Brings TTA (Test Time Functionality) to the `Learner` class. Use `learner.TTA()` instead"
from fastai.torch_core import *
from fastai.basic_train import *
from fastai.basic_train import _loss_func2activ
from fastai.basic_data import DatasetType

__all__ = []


def _tta_only(learn:Learner, ds_type:DatasetType=DatasetType.Valid) -> Iterator[List[Tensor]]:
    "Computes the outputs for several augmented inputs for TTA"
    dl = learn.dl(ds_type)
    ds = dl.dataset
    old = ds.tfms
    augm_tfm = [o for o in learn.data.train_ds.tfms]
    try:
        pbar = master_bar(range(8))
        for i in pbar:
            ds.tfms = augm_tfm
            yield get_preds(learn.model, dl, pbar=pbar, activ=_loss_func2activ(learn.loss_func))[0]
    finally: ds.tfms = old


Learner.tta_only = _tta_only


def _TTA(learn:Learner, beta:float=0.4, ds_type:DatasetType=DatasetType.Valid, with_loss:bool=False) -> Tensors:
    "Applies TTA to predict on `ds_type` dataset."
    preds,y = learn.get_preds(ds_type)
    all_preds = list(learn.tta_only(ds_type=ds_type))
    avg_preds = torch.stack(all_preds).mean(0)
    if beta is None: return preds,avg_preds,y
    else:
        final_preds = preds*beta + avg_preds*(1-beta)
        if with_loss:
            return final_preds, y, calc_loss(final_preds, y, learn.loss_func)
        return final_preds, y


Learner.TTA = _TTA
