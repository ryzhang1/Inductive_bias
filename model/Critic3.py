#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
from Actor1 import init_weights


class Critic(nn.Module):
    def __init__(self, OBS_DIM, ACTION_DIM, TARGET_DIM, RNN_SIZE, FC_SIZE, RNN):
        super().__init__()
        self.OBS_DIM = OBS_DIM
        self.ACTION_DIM = ACTION_DIM
        
        # Q1 architecture
        self.rnn1 = RNN(input_size=OBS_DIM + ACTION_DIM + TARGET_DIM + ACTION_DIM, hidden_size=RNN_SIZE)
        self.l1 = nn.Linear(RNN_SIZE, FC_SIZE)
        self.l2 = nn.Linear(FC_SIZE, FC_SIZE)
        self.l3 = nn.Linear(FC_SIZE, 1)
        
        # Q2 architecture
        self.rnn2 = RNN(input_size=OBS_DIM + ACTION_DIM + TARGET_DIM + ACTION_DIM, hidden_size=RNN_SIZE)
        self.l4 = nn.Linear(RNN_SIZE, FC_SIZE)
        self.l5 = nn.Linear(FC_SIZE, FC_SIZE)
        self.l6 = nn.Linear(FC_SIZE, 1)
        
        self.apply(init_weights)

    def forward(self, x, u, hidden_in1=None, hidden_in2=None, return_hidden=False):
        xu = torch.cat([x, u], dim=2)
        if hidden_in1 is None:
            x1, hidden_out1 = self.rnn1(xu)
        else:
            x1, hidden_out1 = self.rnn1(xu, hidden_in1)
        x1 = F.relu(self.l1(x1))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        
        if hidden_in2 is None:
            x2, hidden_out2 = self.rnn2(xu)
        else:
            x2, hidden_out2 = self.rnn2(xu, hidden_in2)
        x2 = F.relu(self.l4(x2))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        
        if return_hidden:
            return x1, x2, hidden_out1, hidden_out2
        else:
            return x1, x2
    
    def Q1(self, x, u):
        xu = torch.cat([x, u], dim=2)
        x1, _ = self.rnn1(xu)
        x1 = F.relu(self.l1(x1))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        
        return x1

