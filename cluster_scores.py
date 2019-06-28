import sys
import random
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import ujson
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

import matplotlib
import matplotlib.cm as cm
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pylab

latents = ujson.load(open('experiments/tsne_latents.json', 'r'))
latents = np.mean(np.array(latents), axis=1)

print latents.shape

labels = ujson.load(open('experiments/tsne_labels.json', 'r'))
persons = ujson.load(open('experiments/tsne_persons.json', 'r'))
print np.array(persons).shape
print np.array(labels).shape

label_dict = {'M': '0', 'F' : '1'}
pers_dict = {p:i for p,i in zip(set(persons), range(0,len(set(persons))))}
colors = ['red', 'blue']


labels = [label_dict[p] for p in labels]
persons = [pers_dict[p] for p in persons]

labels = np.array(labels)_

persons = np.array(persons)

score = silhouette_score(latents, persons)
print score
