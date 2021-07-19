#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 00:38:47 2021

@author: rakshit
"""
import re
import optim
import argparse
import nn as nn
import numpy as np

from OpenNN import ANN

from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris


# %% Argument parser

# Question: What does this function do?
def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--steps', type=int, default=10000,
                        help='number of epochs/steps')
    args = parser.parse_args()
    return args


# %% Main code

if __name__ == '__main__':

    args = vars(make_args())

    iris_data = load_iris()
    feats = iris_data['data']
    gt = iris_data['target']

    num_samples = feats.shape[0]
    num_classes = gt.max() + 1

    # Venture into ANN's code and answer all questions.
    net = ANN()

    # Venture into nn.linear and nn.sigmoid and answer all questions
    net.operations = [nn.linear(in_c=4, out_c=128),
                      nn.sigmoid(),
                      nn.linear(in_c=128, out_c=num_classes)]

    # Question: Write down the formula for CE loss on paper. Find it's
    # derivate with respect to the input.
    loss_func = nn.CrossEntropyLoss()

    # Question: Write down the formula for gradient descent.
    optimize = optim.SGD(lr=args['lr'])

    samples = np.split(feats, num_samples, axis=0)
    targets = np.split(gt, num_samples, axis=0)
    predict = np.zeros_like(gt)

    for step in range(args['steps']):

        loss_per_epoch = 0

        # Note: In professional packages, we typically do not feed input one
        # sample after another. The forward call is parallelized across
        # multiple samples on a GPU.
        for idx, sample in enumerate(samples):

            # Note: Usually we model data as [N, feats]. However, the math
            # is usually derived in a manner assuming data is [feats, N].
            # We transpose data before feeding into our network.
            # A professional package handles this internally.

            out = net.forward(sample.T)  # Read note above
            loss, loss_grad = loss_func(out, targets[idx])

            loss_per_epoch += loss

            net.loss_grad.append(loss_grad)

            # Accumulate outputs
            predict[idx] = np.argmax(out)

        loss_per_epoch = loss_per_epoch/len(samples)

        c_mat = confusion_matrix(gt, predict)
        c_mat = c_mat.astype('float') / c_mat.sum(axis=1)[:, np.newaxis]

        # Generate gradients to update weights ...
        net.backward()

        # Gradients have been generated, time to update weights!
        optimize(net)

        net.zero_grad()
