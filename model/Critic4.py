#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
from Actor1 import init_weights


class Critic(nn.Module):
    def __init__(self, OBS_DIM, ACTION_DIM, TARGET_DIM, RNN_SIZE, FC_SIZE, RNN):
        super().__init__()
        self.OBS_DIM = OBS_DIM
        self.ACTION_DIM = ACTION_DIM
        
        # Q1 architecture
        self.rnn1 = RNN(input_size=OBS_DIM + ACTION_DIM + TARGET_DIM, hidden_size=RNN_SIZE, num_layers=1)
        self.rnn2 = RNN(input_size=RNN_SIZE + ACTION_DIM, hidden_size=RNN_SIZE, num_layers=1)
        self.l1 = nn.Linear(RNN_SIZE, 1)
        
        # Q2 architecture
        self.rnn3 = RNN(input_size=OBS_DIM + ACTION_DIM + TARGET_DIM, hidden_size=RNN_SIZE, num_layers=1)
        self.rnn4 = RNN(input_size=RNN_SIZE + ACTION_DIM, hidden_size=RNN_SIZE, num_layers=1)
        self.l2 = nn.Linear(RNN_SIZE, 1)
        
        self.apply(init_weights)

    def forward(self, x, u, hidden_in1=None, hidden_in2=None, return_hidden=False):
        if hidden_in1 is None:
            x1, hidden_out11 = self.rnn1(x)
            x1, hidden_out12 = self.rnn2(torch.cat([x1, u], dim=2))
        else:
            hidden_in11, hidden_in12 = hidden_in1
            x1, hidden_out11 = self.rnn1(x, hidden_in11)
            x1, hidden_out12 = self.rnn2(torch.cat([x1, u], dim=2), hidden_in12)
        x1 = self.l1(x1)
        
        if hidden_in2 is None:
            x2, hidden_out21 = self.rnn3(x)
            x2, hidden_out22 = self.rnn4(torch.cat([x2, u], dim=2))
        else:
            hidden_in21, hidden_in22 = hidden_in2
            x2, hidden_out21 = self.rnn3(x, hidden_in21)
            x2, hidden_out22 = self.rnn4(torch.cat([x2, u], dim=2), hidden_in22)
        x2 = self.l2(x2)
        
        if return_hidden:
            return x1, x2, (hidden_out11, hidden_out12), (hidden_out21, hidden_out22)
        else:
            return x1, x2
    
    def Q1(self, x, u):
        x1, _ = self.rnn1(x)
        x1, _ = self.rnn2(torch.cat([x1, u], dim=2))
        x1 = self.l1(x1)
        
        return x1
