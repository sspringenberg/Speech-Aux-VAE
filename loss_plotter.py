import sys
import random

import ujson
import numpy as np

import argparse

import matplotlib
import matplotlib.cm as cm
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pylab

parser = argparse.ArgumentParser(description='VAE Speech')

parser.add_argument('--model-name', type=str, default=None, metavar='S',
                    help='model name (for saving) (default: model type)')

args = parser.parse_args()

losses = ujson.load(open("experiments/{}.json".format('train_losses_'+args.model_name), 'r'))

loss_types= []
loss_values = []
x = range(len(losses))
for t in losses[0]:
    if t =='loss':
        continue
    loss_values.append([epoch[t] for epoch in losses])
    if t =='reconstruction_nll':
        t = 'prediction_nll'
    loss_types.append(t)

for i in range(0,len(loss_types)):
    plt.plot(x, loss_values[i], label=loss_types[i])

plt.legend(loc='upper right')
plt.title('Train Losses')
pylab.savefig('experiments/loss_plot_train.png')

plt.clf()

losses = ujson.load(open("experiments/{}.json".format('test_losses_'+args.model_name), 'r'))

loss_types= []
loss_values = []
x = range(len(losses))
for t in losses[0]:
    if t =='loss':
        continue
    loss_values.append([epoch[t] for epoch in losses])
    if t =='reconstruction_nll':
        t = 'prediction_nll'
    loss_types.append(t)

for i in range(0,len(loss_types)):
    plt.plot(x, loss_values[i], label=loss_types[i])

plt.legend(loc='upper right')

plt.title('Test Losses')
pylab.savefig('experiments/loss_plot_test.png')

