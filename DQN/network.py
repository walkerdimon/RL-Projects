import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb 

class QNet(nn.Module):
    def __init__(self, input_dim):
        super(QNet, self).__init__()

        self.Activation = nn.Tanh()

        self.hidden_1 = nn.Linear(input_dim, 64)

        self.hidden_2 = nn.Linear(64, 64)

        self.output = nn.Linear(64, 1)

    def forward(self, state):
        state = self.hidden_1(state)

        state = self.Activation(state)

        state = self.hidden_2(state)

        state = self.Activation(state)

        Q = self.output(state)

        return Q

