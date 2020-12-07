import numpy as np 
import torch 
import torch.nn as nn
import pdb


class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        self.Activation = nn.Tanh()
        self.hidden_1 = nn.Linear(2, 10).double()
        self.hidden_2 = nn.Linear(10, 10).double()
        self.hidden_3 = nn.Linear(10, 1).double()
        self.std = nn.Parameter(-0.5 * torch.ones(1, dtype=torch.double).double())


    def forward(self, s):
        s = self.hidden_1(s)
        s = self.Activation(s)
        s = self.hidden_2(s)
        s = self.Activation(s)
        mu = self.hidden_3(s)
        std = torch.exp(self.std)

        return (mu, std)



class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.Activation = nn.Tanh()
        self.hidden_1 = nn.Linear(2, 10).double()
        self.hidden_2 = nn.Linear(10, 10).double()
        self.hidden_3 = nn.Linear(10, 1).double()


    def forward(self, s):
        s = self.hidden_1(s)
        s = self.Activation(s)
        s = self.hidden_2(s)
        s = self.Activation(s)
        V = self.hidden_3(s)

        return (V)

