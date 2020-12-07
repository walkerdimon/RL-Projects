import numpy as np 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import pendulum
import pdb 
import argparse
from PPO import PPO



parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=50)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lamb", type=float, default=0.95)

args = parser.parse_args()

def main():

    #initialize environment
    env = pendulum.Pendulum()

    ppo = PPO(args, env)

    for epoch in range(args.epochs):
        print('Training Epoch ' + str(epoch+1) + '/' + str(args.epochs))
        ppo.train()

    ppo.save_model()
    ppo.generate_results()



if __name__ == '__main__':

    main()