#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn

def init_weights(m, mean=0, std=0.1, bias=0):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean, std)
        nn.init.constant_(m.bias, bias)
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, bias)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            else:
                raise ValueError()
                

class Actor(nn.Module):
    def __init__(self, OBS_DIM, ACTION_DIM, TARGET_DIM, RNN_SIZE, FC_SIZE, RNN):
        super().__init__()
        self.OBS_DIM = OBS_DIM
        self.ACTION_DIM = ACTION_DIM
        RNN_SIZE = 220
            
        self.rnn = RNN(input_size=OBS_DIM + ACTION_DIM + TARGET_DIM, hidden_size=RNN_SIZE)
        self.l1 = nn.Linear(RNN_SIZE, ACTION_DIM)
        
        self.apply(init_weights)

    def forward(self, x, hidden_in, return_hidden=True): 
        if hidden_in is None:
            x, hidden_out = self.rnn(x)
        else:
            x, hidden_out = self.rnn(x, hidden_in) 
        x = torch.tanh(self.l1(x))
        
        if return_hidden:
            return x, hidden_out
        else:
            return x

