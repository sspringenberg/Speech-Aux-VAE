This repository contains the code and results of applying two VAE architectures, namely a simple predictive VAE and a multi-timescale auxiliary VAE to speech audio in order to obtain expressive speech representations in an unsupervised manner. The models are implemented in [PyTorch](https://github.com/pytorch/pytorch). The Aux-VAE architecture is further specified in the paper ["Predictive Auxiliary Variational Autoencoder for Representation Learning of Global Speech Characteristics"](), published in the Interspeech 2019 proceedings.

## Models

### Predictive Speech VAE
<p align="center"><img src="./imgs/pred_speech_vae.png" width="600" /></p>

### Multi-Timescale Aux-VAE
<p align="center"><img src="./imgs/m_t_speech_vae.png" width="600" /></p>

## Running the Code

### Preprocessing

In order to preprocess the [LibriSpeech dataset](http://www.openslr.org/12/), please [download](http://www.openslr.org/resources/12/train-clean-100.tar.gz) the train-clean-100 subset first and then run:
```
python preprocess_librispeech.py --librispeech-path={DIR TO VCTK DIRECTORY}
```
with {DIR TO VCTK DIRECTORY} replaced by the path to the LibriSpeech folder. A new folder called librispeech_preprocessed will be created containing preprocessed audio samples.

In order to preprocess the [VCTK dataset](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html), please  [download](https://datashare.is.ed.ac.uk/handle/10283/2651) the dateset first and then run:
```
python preprocess_vctk.py --vctk-path={DIR TO VCTK DIRECTORY}
```
with {DIR TO VCTK DIRECTORY} replaced by the path to the VCTK-Corpus folder. A new folder called vctk_preprocessed will be created containing preprocessed audio samples.

### Running the Model

Models can be trained by running the main.py script. Options include training a simple predicive speech VAE and a multi-timescale auxiliar speech VAE (Aux-VAE)

The predicitve speech VAE can be trained by:
```
python main.py --model-type=vae_l
```

The Aux-VAE model can be trained by:
```
python main.py --model-type=vae_g_l
```

For additional options (such as setting hyperparameters) see:
```
python main.py --h
```

To create mel spectrogram plots and audio recostructions for testing purposes run:
```
python main.py --model-type=TYPE_OF_MODEL --mode=testing
```

To analyse latent representations first run:
```
python main.py --model-type=TYPE_OF_MODEL --mode=analyse-latent
```
To create t-SNE plots then run:
```
python tsne_plotter.py
```
To calculate a silhouette score:
```
python cluster_scores.py
```

To train a linear classifier on top of representations learned by a model in order to perform speaker identification run:
```
python librispeech_speaker_id.py --pretrained-model=TYPE_OF_MODEL
```

### Dependencies
* [Numpy](http://www.numpy.org)
* [Scipy](https://www.scipy.org)
* [LibROSA](https://librosa.github.io/librosa/)
* [SOX](http://sox.sourceforge.net)
* [PyTorch](https://pytorch.org)
* [torchaudio](https://github.com/pytorch/audio)
* [ujson](https://pypi.org/project/ujson/)
* [tqdm](https://github.com/tqdm/tqdm)
* [tensorboardX](https://github.com/lanpa/tensorboardX)
* [Scikit-Learn](http://scikit-learn.org/stable/) (optional, for tsne plots and silhouette scores)
