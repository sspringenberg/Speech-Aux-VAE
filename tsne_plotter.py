import sys
import random
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
from sklearn.manifold import TSNE
import ujson
import numpy as np

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
# pers_dict = {'249':0, '239':1, '276':2, '283':3, '243':4, '254':5, '258':6, '271':7}
pers_dict = {p:i for p,i in zip(set(persons), range(0,len(set(persons))))}
colors = ['red', 'blue']

labels = [label_dict[p] for p in labels]
persons = [pers_dict[p] for p in persons]

labels = np.array(labels)

persons = np.array(persons)

print latents.shape
print labels.shape

l_embedded = TSNE(n_components=2, n_iter=750).fit_transform(latents)

tsne_x = np.array(l_embedded[:,0])
tsne_y = np.array(l_embedded[:,1])

print tsne_x.shape
print tsne_y.shape
print persons.shape

plt.scatter(tsne_x, tsne_y, c=persons, cmap=cm.tab10, label=labels)
plt.legend
plt.title('Speaker ID')
plt.xlabel('t-SNE Projected Latent')
ax = plt.gca()
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
pylab.savefig('experiments/tsne_speaker_id.png')

plt.clf()

plt.scatter(tsne_x, tsne_y, c=labels, cmap=matplotlib.colors.ListedColormap(colors), label=labels)
plt.legend
plt.title('Male / Female')
plt.xlabel('t-SNE Projected Latent')
ax = plt.gca()
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
pylab.savefig('experiments/tsne_male_female.png')
