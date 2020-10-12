import random
import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
from collections import deque
from utils import *
import copy
import os
from os import listdir
from os.path import isfile, join

import discreteaction_pendulum
import pdb

from network import QNet

class Deep_Q_Network(object):
    def __init__(self, args, env):

        #inheret class variables from input arguments
        self.max_episodes = args.max_episodes
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.learning_rate = args.learning_rate
        self.memory_cap = args.memory_cap   
        self.batch_size = args.batch_size   
        self.model_name = args.model_name
        self.C_update = args.C_update
        self.training_step = 0
        self.update_step = 0

        #inheret class variables from pendulum environment 
        self.env = env
        self.num_states = env.num_states
        self.num_actions = env.num_actions
        self.input_dim = 3
        self.memory = deque([])

        #initialize Q and Q_hat Networks
        self.QNet = QNet(self.input_dim)
        self.Q_hat = copy.deepcopy(self.QNet)

        #initialize optimizer
        self.optimizer = torch.optim.Adam(self.QNet.parameters(), lr=self.learning_rate)

        #set up logs
        self.log = { 't' : [0],
                     's' : [],
                     'a' : [],
                     'r' : [],
                        }


    def train_QNet(self):

        #initialize state
        s = self.env.reset()

        done = False
        
        while done == False:

            self.training_step += 1

            #set exploration v exploitation
            self.probability = random.uniform(0,1)

            #take action based on epsilon greedy
            a = e_greedy(self, s)

            #set next state and reward for taking action a
            s1, r, done = self.env.step(a)

            #add state transition to replay memory
            self.memory.append([s[0], s[1], a, r, s1[0], s1[1]])

            #maintain memory size to most recent experiences
            if len(self.memory) > self.memory_cap:
                self.memory.popleft()

            #update network every 'batch size' steps
            if self.training_step % self.batch_size == 0:

                self.update_step += 1

                #zero optimizer gradients
                self.optimizer.zero_grad()


                #get minibatch from experience replay
                self.minibatch = get_minibatch(self)

                #compute target
                y = get_y(self, done)

                #computer Q
                Q = get_Q(self)

                self.loss_QNet = nn.MSELoss()(y, Q)

                self.loss_QNet.backward()

                self.optimizer.step()

                #update Q_hat
                if self.update_step % self.C_update == 0:
                    self.Q_hat = copy.deepcopy(self.QNet)

            #log results
            self.log['t'].append(self.log['t'][-1] + 1)
            self.log['s'].append(s)
            self.log['a'].append(a)
            self.log['r'].append(r)

            #update state
            s = s1






    def save_model(self):
        
        if not os.path.exists('./model_save/' + str(self.model_name) + '/'):
            os.makedirs('./model_save/' + str(self.model_name) + '/')
                
        save_dir = './model_save/' + str(self.model_name) + '/' + str(self.max_episodes) + '_episodes.pt'
            
        torch.save({'episodes': self.max_episodes,
                'QNet_state_dict': self.QNet.state_dict(),
                'QNet_optimizer_state_dict': self.optimizer.state_dict(),
                'log': self.log,
                }, save_dir)


    def generate_results(self):

        if not os.path.isdir('./generated_results/'):
            os.makedirs('./generated_results/')

        DQN_model = torch.load('./model_save/' + str(self.model_name) + '/' + str(self.max_episodes) + '_episodes.pt')
        self.QNet.load_state_dict(DQN_model['QNet_state_dict'])
        self.QNet.eval()
        self.model_log = DQN_model['log']

        #separate data from log
        self.model_log['s'] = np.array(self.model_log['s'])
        self.model_log['r'] = np.array(self.model_log['r'])
        self.theta = self.model_log['s'][:, 0]
        self.thetadot = self.model_log['s'][:, 1]
        self.tau = [self.env._a_to_u(a) for a in self.model_log['a']]
        print('Generating Learning Curve')
        plot_learning_curve(self)
        print('Generating Example Trajectory')
        plot_example_trajectory(self)
        generate_animated_trajectory(self, filename='./generated_results/Animated_Trajectory_' + str(self.max_episodes) + '_Episodes.gif' )
        print('Generating Value Visualization')
        value_visualization(self)
        print('Generating Policy Visualization')
        policy_visualization(self)



    def ablation_study(self):

        if not os.path.isdir('./ablation_study/'):
            os.makedirs('./ablation_study/')

        base_model1 = torch.load('./model_save/base_model/10000_episodes.pt')
        self.base_model1_log = base_model1['log']
        self.base_model1_log['r'] = np.array(self.base_model1_log['r'])
        base_model1_log = np.mean(self.base_model1_log['r'].reshape(-1,10000), axis=1)

        base_model2 = torch.load('./model_save/base_model2/10000_episodes.pt')
        self.base_model2_log = base_model2['log']
        self.base_model2_log['r'] = np.array(self.base_model2_log['r'])
        base_model2_log = np.mean(self.base_model2_log['r'].reshape(-1,10000), axis=1)

        base_model3 = torch.load('./model_save/base_model3/10000_episodes.pt')
        self.base_model3_log = base_model3['log']
        self.base_model3_log['r'] = np.array(self.base_model3_log['r'])
        base_model3_log = np.mean(self.base_model3_log['r'].reshape(-1,10000), axis=1)

        no_replay1 = torch.load('./model_save/no_replay/10000_episodes.pt')
        self.no_replay1_log = no_replay1['log']
        self.no_replay1_log['r'] = np.array(self.no_replay1_log['r'])
        no_replay1_log = np.mean(self.no_replay1_log['r'].reshape(-1,10000), axis=1)

        no_replay2 = torch.load('./model_save/no_replay2/10000_episodes.pt')
        self.no_replay2_log = no_replay2['log']
        self.no_replay2_log['r'] = np.array(self.no_replay2_log['r'])
        no_replay2_log = np.mean(self.no_replay2_log['r'].reshape(-1,10000), axis=1)

        no_replay3 = torch.load('./model_save/no_replay3/10000_episodes.pt')
        self.no_replay3_log = no_replay3['log']
        self.no_replay3_log['r'] = np.array(self.no_replay3_log['r'])
        no_replay3_log = np.mean(self.no_replay3_log['r'].reshape(-1,10000), axis=1)

        no_target1 = torch.load('./model_save/no_target/10000_episodes.pt')
        self.no_target1_log = no_target1['log']
        self.no_target1_log['r'] = np.array(self.no_target1_log['r'])
        no_target1_log = np.mean(self.no_target1_log['r'].reshape(-1,10000), axis=1)

        no_target2 = torch.load('./model_save/no_target2/10000_episodes.pt')
        self.no_target2_log = no_target2['log']
        self.no_target2_log['r'] = np.array(self.no_target2_log['r'])
        no_target2_log = np.mean(self.no_target2_log['r'].reshape(-1,10000), axis=1)

        no_target3 = torch.load('./model_save/no_target3/10000_episodes.pt')
        self.no_target3_log = no_target3['log']
        self.no_target3_log['r'] = np.array(self.no_target3_log['r'])
        no_target3_log = np.mean(self.no_target3_log['r'].reshape(-1,10000), axis=1)

        no_target_no_replay1 = torch.load('./model_save/no_target_no_replay/10000_episodes.pt')
        self.no_target_no_replay1_log = no_target_no_replay1['log']
        self.no_target_no_replay1_log['r'] = np.array(self.no_target_no_replay1_log['r'])
        no_target_no_replay1_log = np.mean(self.no_target_no_replay1_log['r'].reshape(-1,10000), axis=1)

        no_target_no_replay2 = torch.load('./model_save/no_target_no_replay2/10000_episodes.pt')
        self.no_target_no_replay2_log = no_target_no_replay2['log']
        self.no_target_no_replay2_log['r'] = np.array(self.no_target_no_replay2_log['r'])
        no_target_no_replay2_log = np.mean(self.no_target_no_replay2_log['r'].reshape(-1,10000), axis=1)

        no_target_no_replay3 = torch.load('./model_save/no_target_no_replay3/10000_episodes.pt')
        self.no_target_no_replay3_log = no_target_no_replay3['log']
        self.no_target_no_replay3_log['r'] = np.array(self.no_target_no_replay3_log['r'])
        no_target_no_replay3_log = np.mean(self.no_target_no_replay3_log['r'].reshape(-1,10000), axis=1)


        avg_base = (base_model1_log + base_model2_log + base_model3_log) / 3.0

        avg_no_replay = (no_replay1_log + no_replay2_log + no_replay3_log) / 3.0

        avg_no_target = (no_target1_log + no_target2_log + no_target3_log) / 3.0

        avg_no_target_no_replay = (no_target_no_replay1_log + no_target_no_replay2_log + no_target_no_replay3_log) / 3.0

        plt.plot(base_model1_log, label='Base Model')
        plt.plot(no_replay1_log, label='No Replay')
        plt.plot(no_target1_log, label='No Target')
        plt.plot(no_target_no_replay1_log, label='No Replay or Target')
        plt.xlabel('Time Interval')
        plt.ylabel('Discounted Reward')
        plt.legend()
        plt.savefig('./ablation_study/Learning_Curves_' + str(self.max_episodes) + '_Episodes.png' )