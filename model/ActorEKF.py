#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from Actor1 import init_weights


class Actor(nn.Module):
    def __init__(self, EKF_STATE_DIM, ACTION_DIM, FC_SIZE):
        super().__init__()
        self.l1 = nn.Linear(EKF_STATE_DIM, FC_SIZE)
        self.l2 = nn.Linear(FC_SIZE, FC_SIZE)
        self.l3 = nn.Linear(FC_SIZE, ACTION_DIM)
        
        self.apply(init_weights)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return x
