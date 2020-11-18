import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt 

import gridworld
from REINFORCE import REINFORCE

import os 
from os import listdir 
from os.path import isfile, join
import pickle
import pdb
from utils import *

#Input Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--episodes", type=int, default=10000)
parser.add_argument("--learning_rate", type=float, default=0.2)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--model_name", type=str, default='base_model')
parser.add_argument("--base_model", action='store_true')
parser.add_argument("--baseline_causality", action='store_true')
parser.add_argument("--baseline_importance", action='store_true')
parser.add_argument("--causality_importance", action='store_true')
parser.add_argument("--baseline_causality_importance", action='store_true')

args = parser.parse_args()

def main():

    #Initialize Gridworld Environment
    env = gridworld.GridWorld()

    #Initialize REINFORCE algorithm
    reinforce = REINFORCE(args, env)

    #run reinforce algorithm
    for episode in range(args.episodes):

        print('Episode ' + str(episode+1) + '/' + str(args.episodes))
        reinforce.train()

    #save results once training is finished
    reinforce.save_model()

    #generate results
    #plot_learning_curve(args)
    #generate_policy(args, env)



if __name__ == '__main__':

    main()