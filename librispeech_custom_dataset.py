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
    dir = os.path.join(dir, "train-clean-100")
    dir = os.path.expanduser(dir)
    for person in sorted(os.listdir(dir)):
        pers_dir = os.path.join(dir, person)
        pers_dir = os.path.expanduser(pers_dir)
        for session in sorted(os.listdir(pers_dir)):            
            flac_dir = os.path.join(pers_dir, session)
            flac_dir = os.path.expanduser(flac_dir)
            for audiofile in sorted(os.listdir(flac_dir)):
                if not audiofile.endswith(".flac"):
                    continue            
                audios.append(os.path.expanduser(os.path.join(flac_dir, audiofile)))
    return audios

class LibriSpeech(data.Dataset):
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
    
    def __init__(self, root, downsample=True, transform=None, target_transform=None, dev_mode=False, preprocessed=False, person_filter=None, filter_mode = 'exclude', max_len=201, split='train'):
        self.person_filter = person_filter
        self.filter_mode = filter_mode
        self.root = os.path.expanduser(root)
        self.downsample = downsample
        self.transform = transform
        self.target_transform = target_transform
        self.dev_mode = dev_mode
        self.num_samples = 0
        self.max_len = max_len
        self.split = split
        
        if preprocessed:
            self.root_dir = os.path.expanduser('librispeech_preprocessed/')
            if self.split == 'train':
                self.data_paths = os.listdir(os.path.join(self.root_dir,'train'))
                self.root_dir = os.path.join(self.root_dir,'train/')
            elif self.split == 'test':
                self.data_paths = os.listdir(os.path.join(self.root_dir,'test'))
                self.root_dir = os.path.join(self.root_dir,'test/')
            
            if person_filter:
                if self.filter_mode == 'include':                    
                    self.data_paths = [sample for sample in self.data_paths if any(sample.startswith(pers+'-') for pers in self.person_filter)]
                elif self.filter_mode == 'exclude':
                    self.data_paths = [sample for sample in self.data_paths if not any(sample.startswith(pers+'-') for pers in self.person_filter)]
            
            self.num_samples = len(self.data_paths)
            
        else:            
            paths = make_manifest(self.root)
            os.mkdir('librispeech_preprocessed')
            os.mkdir('librispeech_preprocessed/train')
            os.mkdir('librispeech_preprocessed/test')
            
            test_splits = open("librispeech_splits/test_split.txt")
            train_splits = open("librispeech_splits/train_split.txt")
            split_reader = csv.reader(test_splits)
            test_data = [r[0] for r in split_reader]
            split_reader = csv.reader(train_splits)
            train_data = [r[0] for r in split_reader]
            
            with open(os.path.join(self.root,"SPEAKERS.TXT")) as csvfile:                
                csvreader = csv.reader(csvfile, delimiter='|')
                for i in range(12):
                    next(csvreader)
                rows = [r for r in csvreader]
                dict = {x[0].strip():[x[1].strip()] for x in rows}
                for z, path in enumerate(paths):              
                    
                    keyword = 'train-clean-100/'
                    before_keyword, keyword, after_keyword = path.partition(keyword)
                    before_keyword, keyword, after_keyword = after_keyword.partition('/')
                    pers = before_keyword
                    before_keyword, keyword, after_keyword = after_keyword.partition('/')
                    before_keyword, keyword, after_keyword = after_keyword.partition('.flac')
                    
                    sig = read_audio(path)
                    if self.transform is not None:
                        sig = self.transform(sig[0])
                        
                    else:
                        sig = sig[0]
                    
                    try:
                        data = (sig.tolist(), dict[pers] + [pers])
                        if before_keyword in train_data:
                            ujson.dump(data,open("librispeech_preprocessed/train/{}.json".format(before_keyword), 'w'))
                        elif before_keyword in test_data:
                            ujson.dump(data,open("librispeech_preprocessed/test/{}.json".format(before_keyword), 'w'))
                        if z % 100 == 0:
                            print "{} iterations".format(z)
                        self.train_data_paths = os.listdir(os.path.expanduser('librispeech_preprocessed/train/'))
                        self.test_data_paths = os.listdir(os.path.expanduser('librispeech_preprocessed/test/'))
                    except:
                        continue
            
            self.train_data_paths = os.listdir(os.path.expanduser('librispeech_preprocessed/train/'))
            self.test_data_paths = os.listdir(os.path.expanduser('librispeech_preprocessed/test/'))
            self.num_samples = len(self.train_data_paths)
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
