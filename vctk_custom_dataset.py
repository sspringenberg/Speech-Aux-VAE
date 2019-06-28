import torch.utils.data as data
import csv
import string
import unicodedata
import re
import os
import os.path
import sys
import shutil
import errno
import torch
import torchaudio
import ujson
from audio_utils import griffinlim
from audio_utils import read_audio
import librosa
import numpy as np
from torchaudio import datasets, transforms, save
import torch.nn.functional as F

def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def make_manifest(dir):
    audios = []
    dir = os.path.join(dir, "wav48")
    dir = os.path.expanduser(dir)
    for person in sorted(os.listdir(dir)):
        wav_dir = os.path.join(dir, person)
        wav_dir = os.path.expanduser(wav_dir)
        for audiofile in sorted(os.listdir(wav_dir)):
            if not audiofile.endswith(".wav"):
                continue            
            audios.append(os.path.expanduser(os.path.join(wav_dir, audiofile)))
    return audios

class VCTK(data.Dataset):
    """`VCTK <http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html>`_ Dataset.
    `alternate url <http://datashare.is.ed.ac.uk/handle/10283/2651>`
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.Scale``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        dev_mode(bool, optional): if true, clean up is not performed on downloaded
            files.  Useful to keep raw audio and transcriptions.
    """
    raw_folder = '/wavs'
    processed_folder = '/processed'

    def __init__(self, root, downsample=True, transform=None, target_transform=None, dev_mode=False, preprocessed=False, person_filter=None, filter_mode = 'exclude', max_len=201):
        self.person_filter = person_filter
        self.filter_mode = filter_mode
        self.root = os.path.expanduser(root)
        self.downsample = downsample
        self.transform = transform
        self.target_transform = target_transform
        self.dev_mode = dev_mode
        self.num_samples = 0
        self.max_len = max_len
        
        if preprocessed:
            self.root_dir = os.path.expanduser('vctk_preprocessed/')
            self.data_paths = os.listdir(self.root_dir)
            
            if person_filter:
                if self.filter_mode == 'include':                    
                    self.data_paths = [sample for sample in self.data_paths if any(pers in sample for pers in self.person_filter)]
                elif self.filter_mode == 'exclude':
                    self.data_paths = [sample for sample in self.data_paths if not any(pers in sample for pers in self.person_filter)]
            
            self.num_samples = len(self.data_paths)
            
        else:            
            paths = make_manifest(self.root)
            os.mkdir('vctk_preprocessed/')
            with open(os.path.join(self.root,"speaker-info.txt")) as csvfile:
                csvreader = csv.reader(csvfile, delimiter=' ')
                next(csvreader)
                rows = [r for r in csvreader]
                dict = {x[0]:[x[4], x[2], x[8]] for x in rows}
                for z, path in enumerate(paths):    
                    
                    keyword = 'wav48/'
                    befor_keyowrd, keyword, after_keyword = path.partition(keyword)
                    pers = after_keyword[1:4]
                    
                    sig = read_audio(path)
                    if self.transform is not None:
                        sig = self.transform(sig[0])
                    else:
                        sig = sig[0]
                    try:
                        self.data = (sig.tolist(), dict[pers] + [pers])
                        ujson.dump(self.data,open("vctk_preprocessed/{}.json".format(after_keyword[5:13]), 'w'))
                        if z % 100 == 0:
                            print "{} iterations".format(z)
                        self.data_paths = os.listdir(os.path.expanduser('vctk_preprocessed/'))
                    except:
                        continue
            
            self.data_paths = os.listdir(os.path.expanduser('vctk_preprocessed/'))
            self.num_samples = len(self.data_paths)
            print "{} samples processed".format(self.num_samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """        
        # print self.data_paths[0]
        
        audio, label = ujson.load(open(self.root_dir+self.data_paths[index], 'r'))
        audio = torch.from_numpy(np.array(audio)).float()
        
        if audio.size(1) < self.max_len:
            audio = F.pad(audio, (0, self.max_len-audio.size(1)), "constant", -80.)
        elif audio.size(1) > self.max_len:
            audio = audio[:, :self.max_len]
        
        return audio, label

    def __len__(self):
        return self.num_samples
