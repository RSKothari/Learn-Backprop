#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 00:29:24 2021

@author: rakshit
"""

import numpy as np


class Module():
    # Question: What is the purpose of this class?
    def __init__(self, ):
        self.grad = []
        self.eval = True  # All layers default on eval mode
        self.w_update = []
        self.stored_input = []

    def zero_grad(self, ):
        self.grad = []
        self.stored_input = []


class linear(Module):
    def __init__(self, in_c, out_c, bias=True):
        # Question: What does the __init__ function do?
        super(linear, self).__init__()
        self.type = 'linear'
        self.eval = False
        self.w = [np.random.rand(out_c, in_c),  # Weight
                  np.random.rand(out_c, 1)]     # Bias

    def forward(self, x):
        '''
        Parameters
        ----------
        x : numpy array of shape [feats, 1]
            Input feature vector.
        '''
        self.stored_input.append(x)
        return self.w[0].dot(x) + self.w[1]

    def grad_func(self, x):
        # Gradient function for a linear layer
        return self.w[0]

    def backward(self, prev_grad):

        w_grad, b_grad = [], []

        # Since we pass every sample separately, we must compute the gradient
        # for each samples separately. Loop over the stored inputs.
        for idx, ele in enumerate(self.stored_input):
            self.grad.append(self.grad_func(ele))

            w_grad.append(prev_grad[idx].dot(ele.T))
            b_grad.append(prev_grad[idx])

        # Average weight gradient across all samples
        w_grad = np.mean(np.stack(w_grad, axis=0), axis=0)
        assert w_grad.shape == self.w[0].shape, 'W grad shape does not match'

        # Average bias gradient across all samples
        b_grad = np.mean(np.stack(b_grad, axis=0), axis=0)
        assert b_grad.shape == self.w[1].shape, 'B grad shape does not match'

        self.w_update = [w_grad, b_grad]


def sigmoid_(x):
    return 1/(1 + np.exp(-x))


def softmax_(x):
    return np.exp(x)/np.sum(np.exp(x))


class sigmoid(Module):
    def __init__(self, ):
        super(sigmoid, self).__init__()
        self.type = 'act_func'
        pass

    def forward(self, x):
        self.stored_input.append(x)
        return sigmoid_(x)

    def grad_func(self, x):
        # Question: Find the derivative of sigmoid.
        return sigmoid_(x)*sigmoid_(1-x)

    def backward(self, prev_grad):
        self.grad = [self.grad_func(ele) for ele in self.stored_input]
        pass


class MSE_loss():
    def __init__(self, ):
        self.type = 'loss_func'
        pass

    def grad_func(self, y_pred, y_gt):
        # Question: Find the derivate of MSEloss
        out = y_pred - y_gt
        return out

    def forward(self, y_pred, y_gt):
        out = 0.5*np.mean(np.power(y_pred - y_gt, 2))
        return out

    def __call__(self, y_pred, y_gt):
        # Question: What does __call__ do?
        return self.forward(y_pred, y_gt), self.grad_func(y_pred, y_gt)


class CrossEntropyLoss():
    def __init__(self, ):
        self.type = 'loss_func'
        pass

    def grad_func(self, y_pred, y_gt):
        out = softmax_(y_pred)
        out[y_gt] = out[y_gt] - 1
        return out

    def forward(self, y_pred, y_gt):
        log_softmax = np.log(softmax_(y_pred))
        return -log_softmax[y_gt]

    def __call__(self, y_pred, y_gt):
        return self.forward(y_pred, y_gt), self.grad_func(y_pred, y_gt)
