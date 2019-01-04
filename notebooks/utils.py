from functools import partial
from pathlib import Path
from multiprocessing import Pool
import os
import shutil
import numpy as np
import pandas as pd
import librosa
from scipy.io import wavfile
from tqdm import tqdm_notebook as tqdm
import torch.nn.functional as F
from fastai.basic_data import DatasetType


def read_file(filename, path='', sample_rate=None, trim=False):
    ''' Reads in a wav file and returns it as an np.float32 array in the range [-1,1] '''
    filename = Path(path) / filename
    file_sr, data = wavfile.read(filename)
    if data.dtype == np.int16:
        data = np.float32(data) / np.iinfo(np.int16).max
    elif data.dtype != np.float32:
        raise OSError('Encounted unexpected dtype: {}'.format(data.dtype))
    if sample_rate is not None and sample_rate != file_sr:
        if len(data) > 0:
            data = librosa.core.resample(data, file_sr, sample_rate, res_type='kaiser_fast')
        file_sr = sample_rate
    if trim and len(data) > 1:
        data = librosa.effects.trim(data, top_db=40)[0]
    return data, file_sr


def write_file(data, filename, path='', sample_rate=44100):
    ''' Writes a wav file to disk stored as int16 '''
    filename = Path(path) / filename
    if data.dtype == np.int16:
        int_data = data
    elif data.dtype == np.float32:
        int_data = np.int16(data * np.iinfo(np.int16).max)
    else:
        raise OSError('Input datatype {} not supported, use np.float32'.format(data.dtype))
    wavfile.write(filename, sample_rate, int_data)


def load_audio_files(path, filenames=None, sample_rate=None, trim=False):
    '''
    Loads in audio files and resamples if necessary.
    
    Args:
        path (str or PosixPath): directory where the audio files are located
        filenames (list of str): list of filenames to load. if not provided, load all 
                                 files in path
        sampling_rate (int): if provided, audio will be resampled to this rate
        trim (bool): 
    
    Returns:
        list of audio files as numpy arrays, dtype np.float32 between [-1, 1]
    '''
    path = Path(path)
    if filenames is None:
        filenames = sorted(list(f.name for f in path.iterdir()))
    files = []
    for filename in tqdm(filenames, unit='files'):
        data, file_sr = read_file(filename, path, sample_rate=sample_rate, trim=trim)
        files.append(data)
    return files
    
        
def _resample(filename, src_path, dst_path, sample_rate=16000, trim=True):
    data, sr = read_file(filename, path=src_path, sample_rate=sample_rate, trim=trim)
    write_file(data, filename, path=dst_path, sample_rate=sample_rate)
    

def resample_path(src_path, dst_path, **kwargs):
    transform_path(src_path, dst_path, _resample, **kwargs)    
    

def _to_mono(filename, dst_path):
    data, sr = read_file(filename)
    if len(data.shape) > 1:
        data = librosa.core.to_mono(data.T) # expects 2,n.. read_file returns n,2
    write_file(data, dst_path/filename.name, sample_rate=sr)


def convert_to_mono(src_path, dst_path, processes=None):
    src_path, dst_path = Path(src_path), Path(dst_path)
    os.makedirs(dst_path, exist_ok=True)
    filenames = list(src_path.iterdir())
    convert_fn = partial(_to_mono, dst_path=dst_path)
    with Pool(processes=processes) as pool:
        with tqdm(total=len(filenames), unit='files') as pbar:
            for _ in pool.imap_unordered(convert_fn, filenames):
                pbar.update()
                
                
def transform_path(src_path, dst_path, transform_fn, fnames=None, processes=None, delete=False, **kwargs):
    src_path, dst_path = Path(src_path), Path(dst_path)
    if dst_path.exists() and delete:
        shutil.rmtree(dst_path)
    os.makedirs(dst_path, exist_ok=True)
    
    _transformer = partial(transform_fn, src_path=src_path, dst_path=dst_path, **kwargs)
    if fnames is None:
        fnames = [f.name for f in src_path.iterdir()]
    with Pool(processes=processes) as pool:
        with tqdm(total=len(fnames), unit='files') as pbar:
            for _ in pool.imap_unordered(_transformer, fnames):
                pbar.update()


class RandomPitchShift():
    def __init__(self, sample_rate=22050, max_steps=3):
        self.sample_rate = sample_rate
        self.max_steps = max_steps
    def __call__(self, x):
        n_steps = np.random.uniform(-self.max_steps, self.max_steps)
        x = librosa.effects.pitch_shift(x, sr=self.sample_rate, n_steps=n_steps)
        return x


def _make_transforms(filename, src_path, dst_path, tfm_fn, sample_rate=22050, n_tfms=5):
    data, sr = read_file(filename, path=src_path)
    fn = Path(filename)
    # copy original file 
    new_fn = fn.stem + '_00.wav'
    write_file(data, new_fn, path=dst_path, sample_rate=sample_rate)
    # make n_tfms modified files
    for i in range(n_tfms):
        new_fn = fn.stem + '_{:02d}'.format(i+1) + '.wav'
        if not (dst_path/new_fn).exists():
            x = tfm_fn(data)
            write_file(x, new_fn, path=dst_path, sample_rate=sample_rate)


def pitch_shift_path(src_path, dst_path, max_steps, sample_rate, n_tfms=5):
    pitch_shifter = RandomPitchShift(sample_rate=sample_rate, max_steps=max_steps)
    transform_path(src_path, dst_path, _make_transforms, 
                   tfm_fn=pitch_shifter, sample_rate=sample_rate, n_tfms=n_tfms)
    
    
def rand_pad_crop(signal, pad_start_pct=0.1, crop_end_pct=0.5):
    r_pad, r_crop = np.random.rand(2)
    pad_start = int(pad_start_pct * r_pad * signal.shape[0])
    crop_end  = int(crop_end_pct * r_crop * signal.shape[0]) + 1
    return F.pad(signal[:-crop_end], (pad_start, 0), mode='constant')


def get_transforms(min_len=2048):
    def _train_tfm(x):
        x = rand_pad_crop(x)
        if x.shape[0] < min_len:
            x = F.pad(x, (0, min_len - x.shape[0]), mode='constant')
        return x
    
    def _valid_tfm(x):
        if x.shape[0] < min_len:
            x = F.pad(x, (0, min_len - x.shape[0]), mode='constant')
        return x
  
    return [_train_tfm],[_valid_tfm]


def save_submission(learn, filename, tta=False):
    fnames = [Path(f).name for f in learn.data.test_ds.x.items]
    get_predsfn = learn.TTA if tta else learn.get_preds
    preds = get_predsfn(ds_type=DatasetType.Test)[0]
    top_3 = np.array(learn.data.classes)[np.argsort(-preds, axis=1)[:, :3]]
    labels = [' '.join(list(x)) for x in top_3]
    df = pd.DataFrame({'fname': fnames, 'label': labels})
    df.to_csv(filename, index=False)
    return df


def precision(y_pred, y_true, thresh:float=0.2, eps:float=1e-9, sigmoid:bool=True):
    "Computes the f_beta between preds and targets"
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred>thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum(dim=1)
    prec = TP/(y_pred.sum(dim=1)+eps)
    return prec.mean()


def recall(y_pred, y_true, thresh:float=0.2, eps:float=1e-9, sigmoid:bool=True):
    "Computes the f_beta between preds and targets"
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred>thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum(dim=1)
    rec = TP/(y_true.sum(dim=1)+eps)
    return rec.mean()
