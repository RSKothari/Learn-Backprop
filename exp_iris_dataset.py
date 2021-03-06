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

    # Create a new network instance
    net = ...

    # Define your own layers and activation units    
    net.operations = []

    loss_func = nn.CrossEntropyLoss()

    # Define an optimizer function
    optimize = ...

    samples = np.split(feats, num_samples, axis=0)
    targets = np.split(gt, num_samples, axis=0)
    predict = np.zeros_like(gt)

    for step in range(args['steps']):

        loss_per_epoch = 0

        for idx, sample in enumerate(samples):

            # Explain why we transpose the input sample
            out = net.forward(sample.T) 
            
            # Compute loss and the gradient using CrossEntropyLoss function
            loss, loss_grad = ...

            loss_per_epoch += loss

            net.loss_grad.append(loss_grad)

            # Accumulate outputs
            predict[idx] = np.argmax(out)

        loss_per_epoch = loss_per_epoch/len(samples)

        # Generate a confusion matrix based on performance

        # Generate gradients to update weights ...

        # Gradients have been generated, time to update weights!

        # Remove out accumulated gradients
        
    # Plot training performance
