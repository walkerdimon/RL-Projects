import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

import discreteaction_pendulum
from network import QNet
from DQN import Deep_Q_Network

import os
from os import listdir
from os.path import isfile, join
import pickle

import pdb 


#Input Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train", action='store_true')
parser.add_argument("--test", action='store_true')
parser.add_argument("--ablation", action='store_true')
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--epsilon", type=float, default=0.1)
parser.add_argument("--learning_rate", type=float, default=0.00025)
parser.add_argument("--memory_cap", type=int, default = 10000)
parser.add_argument("--batch_size", type=int, default = 32)
parser.add_argument("--max_episodes", type=int, default=10000)
parser.add_argument("--model_name", type=str, default='base_model')
parser.add_argument("--C_update", type=int, default=5)

args = parser.parse_args()


def main():

    env = discreteaction_pendulum.Pendulum()

    #Initialize the DQL Model
    DQN = Deep_Q_Network(args, env)


    if args.train:

        #Train DQL Model
        for episode in range(args.max_episodes):
        
            print('Episode ' + str(episode+1) + '/' + str(args.max_episodes))
            DQN.train_QNet()

        print('Training Complete. Saving Model.')
        #Save Model
        DQN.save_model()
        print('Model Saved. Done.')


    if args.test:
        #generate results
        print('Generating Results...')
        DQN.generate_results()

        print('Done.')


    if args.ablation:
        print('Generating Ablation Study...')
        DQN.ablation_study()

        print('Done.')



if __name__ == '__main__':

    main()


