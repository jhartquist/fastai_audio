from functools import partial
from pathlib import Path
import os
from multiprocessing import Pool
import numpy as np
import librosa
from scipy.io import wavfile
from tqdm import tqdm_notebook as tqdm


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
    
    
def resample_path(src_path, dst_path, sample_rate=16000, trim=True):
    ''' Resamples a folder of wav files into a new folder at the given sample_rate '''
    src_path, dst_path = Path(src_path), Path(dst_path)
    os.makedirs(dst_path, exist_ok=True)
    filenames = list(src_path.iterdir())
    for filename in tqdm(filenames, unit="files"):
        data, file_sr = read_file(filename, sample_rate=sample_rate, trim=trim)
        write_file(data, dst_path/filename.name, sample_rate=sample_rate)
        

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