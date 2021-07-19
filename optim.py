#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 17:25:37 2021

@author: rakshit
"""


class SGD():
    def __init__(self, lr=1e-3):
        self.lr = lr

    def __call__(self, model):
        for op in model.operations:
            if not op.eval:
                for idx, dE_dW in enumerate(op.w_update):
                    op.w[idx] = op.w[idx] - self.lr*dE_dW
