# fastai_audio

This is an experimental and unofficial module add-on for the new [`fastai v1`](https://github.com/fastai/fastai) library.  It adds the capability of audio classification to fastai by loading raw audio files and generating spectrograms on the fly.  Please check out the notebooks directory for usage examples.

The accompanying article can be found here:
[Audio Classification using FastAI and On-the-Fly Frequency Transforms](https://medium.com/@johnhartquist/audio-classification-using-fastai-and-on-the-fly-frequency-transforms-4dbe1b540f89)

##### Related Links
* [fastai v1 docs](https://docs.fastai.com)

##### Note:
The [`fastai`](https://github.com/fastai/fastai) library is currently being developed rapidly, so this repo may quickly go out of date with new versions.

#### Dependencies
* python 3.6
* fastai 1.0.28
* librosa 0.6.2 

This repo was also heavily inspired by [torchaudio](http://pytorch.org/audio/), especially [`transforms.py`](http://pytorch.org/audio/_modules/torchaudio/transforms.html).
