#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from Actor1 import init_weights


class Actor(nn.Module):
    def __init__(self, OBS_DIM, ACTION_DIM, TARGET_DIM, RNN_SIZE, FC_SIZE, RNN):
        super().__init__()
        self.OBS_DIM = OBS_DIM
        self.ACTION_DIM = ACTION_DIM
       
        self.rnn = RNN(input_size=OBS_DIM + ACTION_DIM + TARGET_DIM, hidden_size=RNN_SIZE)
        self.l1 = nn.Linear(RNN_SIZE, FC_SIZE)
        self.l2 = nn.Linear(FC_SIZE, FC_SIZE)
        self.l3 = nn.Linear(FC_SIZE, ACTION_DIM)
        
        self.apply(init_weights)

    def forward(self, x, hidden_in, return_hidden=True): 
        if hidden_in is None:
            x, hidden_out = self.rnn(x)
        else:
            x, hidden_out = self.rnn(x, hidden_in)
          
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        if return_hidden:
            return x, hidden_out
        else:
            return x

