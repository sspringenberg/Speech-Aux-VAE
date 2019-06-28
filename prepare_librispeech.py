import argparse
import torch
import torchaudio
from torchaudio import transforms, save
import numpy as np
import ujson
from librispeech_custom_dataset import LibriSpeech
import librosa
from audio_utils import griffinlim

import audio_utils as prepro

parser = argparse.ArgumentParser(description='Prepare LibriSpeech Dataset')

parser.add_argument('--librispeech-path', type=str, metavar='S', required=True, help='(path to LibriSpeech-Corpus)')

args = parser.parse_args()

transfs = transforms.Compose([
        prepro.DB_Spec(sr = 11025, n_fft=400,hop_t=0.010,win_t=0.025, max_len=80000)
        ])

dataset = LibriSpeech(root = args.librispeech_path, transform = transfs)
