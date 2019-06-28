import argparse
import torch
import torchaudio
from torchaudio import transforms, save
import numpy as np
import ujson
from vctk_custom_dataset import VCTK
import librosa
from audio_utils import griffinlim

import audio_utils as prepro

parser = argparse.ArgumentParser(description='Prepare VCTK Dataset')

parser.add_argument('--vctk-path', type=str, metavar='S', required=True, help='(path to VCTK-Corpus)')

args = parser.parse_args()

transfs = transforms.Compose([
        prepro.DB_Spec(sr = 11025, n_fft=400,hop_t=0.010,win_t=0.025)
        ])

dataset = VCTK(root = args.vctk_path, transform = transfs)
