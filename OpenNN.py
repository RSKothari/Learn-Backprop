#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 13:29:03 2021

@author: rakshit
"""

from nn import Module

"""
forward function:
    Forward neural network operation. The network stores all inputs that passed
    through a layer. These inputs are used to compute the gradient and can be
    freed using the zero_grad() function

backward function:
    Computes the gradient for each input saved in the layer. Gradients must be
    freed to ensure memory does not grow.

zero_grad function:
    A function which frees up gradients, saved input arrays and

Useful resources:
    https://www.ics.uci.edu/~pjsadows/notes.pdf
    https://www.jasonosajima.com/backprop
"""


class ANN(Module):
    # Question: Why do we wrap ANN(Module)?
    def __init__(self, ):
        super(ANN, self).__init__()
        # Question: What does the "super" do?
        self.loss_grad = []
        self.operations = []

    def forward(self, x):
        # Question: What does the attribute "operations" hold?
        for layer in self.operations:
            x = layer.forward(x)
        return x

    def backward(self):
        # To the students, if you are confused about backprop derivation,
        # please consider going over this article:
        # https://www.jasonosajima.com/backprop

        # Start by the gradient generated at the loss function
        prev_grad = self.loss_grad

        # Iterate through every layer in the network.
        # Question: What does "idx" and "layer" hold? What is the importance of
        # the "enumerate" function?
        for idx, layer in enumerate(reversed(self.operations)):

            # Find the gradient at "layer"
            layer.backward(prev_grad)
            for idx, ele in enumerate(layer.grad):
                if layer.type == 'act_func':
                    prev_grad[idx] = prev_grad[idx]*ele
                else:
                    prev_grad[idx] = ele.T.dot(prev_grad[idx])

            # Question: What is the difference between doing mat_A*mat_B and
            # mat_A.dot(mat_B)? Here, mat_A and mat_B are two matrices.

    def zero_grad(self, ):
        self.loss_grad = []
        for layer in self.operations:
            layer.zero_grad()
