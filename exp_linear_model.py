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


# %% Argument parser

# Question: What does this function do?
def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--steps', type=int, default=10000,
                        help='number of epochs/steps')
    args = parser.parse_args()
    return args

# %% Useful helperfunctions


def generate_points(num_pts=1000, add_noise=True, mode='3D'):

    #  Generate random points along a line
    num_input_dims = int(re.findall(r'\d+', mode)[0]) - 1

    random_slope = 5*np.random.rand(num_input_dims, ) - 2.5  # Random slope
    random_intercept = 20*np.random.rand(1, ) - 10  # Random intercept

    ip_pts = 100*np.random.rand(num_pts, num_input_dims) - 50

    op_gt = np.dot(ip_pts, random_slope) + random_intercept

    if add_noise:
        # Question: What is the difference between np.random.rand and
        # np.random.normal
        ip_noise = np.random.normal(0, 20, (num_pts, num_input_dims))
        op_noise = np.random.normal(0, 20, (num_pts, ))
    else:
        ip_noise = 0
        op_noise = 0

    gt_params = {'slope': random_slope,
                 'intercept': random_intercept}

    return (ip_pts + ip_noise, op_gt + op_noise, gt_params)


# %% Main code

if __name__ == '__main__':

    args = vars(make_args())

    feats, gt, gt_params = generate_points(add_noise=True)
    num_samples = feats.shape[0]

    # Venture into ANN's code and answer all questions.
    net = ANN()

    # Venture into nn.linear and nn.sigmoid and answer all questions
    net.operations = [nn.linear(in_c=2, out_c=12),
                      nn.sigmoid(),
                      nn.linear(in_c=12, out_c=1)]

    # Question: Write down the formula for MSE loss on paper. Find it's
    # derivate with respect to the input.
    loss_func = nn.MSE_loss()

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
            predict[idx, ...] = out

        loss_per_epoch = loss_per_epoch/len(samples)
        print(loss_per_epoch)

        # Generate gradients to update weights ...
        net.backward()

        # Gradients have been generated, time to update weights!
        optimize(net)

        net.zero_grad()
