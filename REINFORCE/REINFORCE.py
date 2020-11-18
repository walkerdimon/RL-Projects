import numpy as np
import torch
import pdb 
import os 
from os import listdir 
from os.path import isfile, join
from utils import *

class REINFORCE(object):
    def __init__(self, args, env):

        #inheret clss variables from input arguments
        self.episodes = args.episodes
        self.gamma = args.gamma
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.model_name = args.model_name

        self.base_model = args.base_model
        self.baseline_causality = args.baseline_causality
        self.baseline_importance = args.baseline_importance
        self.baseline_causality_importance = args.baseline_causality_importance
        self.causality_importance = args.causality_importance


        #inheret class variables from environment variables
        self.env = env
        self.num_states = env.num_states
        self.num_actions = env.num_actions


        #initialize weight matrix
        rg = np.random.default_rng()
        self.weights = torch.tensor(rg.standard_normal(size=(self.num_states, self.num_actions)), requires_grad=True)

        #initialize optimizer
        #self.optimizer = torch.optim.SGD([self.weights], lr=self.learning_rate) #--OG MODEL USES SGD
        self.optimizer = torch.optim.Adam([self.weights], lr=self.learning_rate) #--comparison model uses Adam

        #set up logs
        self.steps = 0
        self.rewards = 0
        self.log = {'steps': [], 
                    'reward': []}



    def train(self):

        with torch.no_grad():

            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_causality_rewards = []
            batch_log_policy = []
            total_average_reward = 0

            for i in range(self.batch_size):

                #initialize the state
                s = self.env.reset()

                trajectory_states = []
                trajectory_actions = []
                trajectory_rewards = []
                trajectory_log_policy = []


                done = False
                
                while done ==False:

                    #get action from current state
                    logits = self.weights[s]
                    distribution = torch.distributions.categorical.Categorical(logits=logits)
                    a = distribution.sample().item()

                    #update state & compute reward
                    s1, r, done = self.env.step(a)

                    trajectory_states.append(s)
                    trajectory_actions.append(a)
                    trajectory_rewards.append(r)
                    a = torch.tensor(a, dtype=torch.float64)
                    trajectory_log_policy.append(distribution.log_prob(a))

                    #if episode continues, update state
                    s = s1


                    #if episode finished, update policy before termination
                    if done == True:
                        batch_states.extend(trajectory_states)
                        batch_actions.extend(trajectory_actions)
                        batch_rewards.extend([np.sum(trajectory_rewards)] * len(trajectory_rewards))
                        causality_list = [sum(trajectory_rewards[i:]) for i in range(len(trajectory_rewards))]
                        batch_causality_rewards.extend(causality_list)
                        batch_log_policy.extend(trajectory_log_policy)
                        total_average_reward += np.sum(trajectory_rewards)
            
            total_average_reward /= self.batch_size


        #once batch is finished, update weights & log reward for episode
        self.steps += self.batch_size * self.env.max_num_steps
        self.log['steps'].append(self.steps)
        self.log['reward'].append(total_average_reward)

        batch_states = torch.tensor(batch_states, requires_grad=False)
        batch_actions = torch.tensor(batch_actions, requires_grad=False)
        batch_log_policy = torch.tensor(batch_log_policy, requires_grad=False)
        batch_rewards = torch.tensor(batch_rewards, requires_grad=False, dtype=torch.float64)
        batch_causality_rewards = torch.tensor(batch_causality_rewards, requires_grad=False, dtype=torch.float64)


        if self.base_model:
            logits = self.weights[batch_states]
            distribution = torch.distributions.categorical.Categorical(logits=logits)
            self.optimizer.zero_grad()
            loss = - (self.env.max_num_steps * distribution.log_prob(batch_actions) * batch_rewards).mean()
            loss.backward()
            self.optimizer.step()


        if self.baseline_causality:
            baseline_shift = []
            batch_baseline = torch.tensor(np.full(len(batch_causality_rewards), total_average_reward), requires_grad=False)
            baseline_shift = batch_causality_rewards - batch_baseline

            logits = self.weights[batch_states]
            distribution = torch.distributions.categorical.Categorical(logits=logits)
            self.optimizer.zero_grad()
            loss = - (self.env.max_num_steps * distribution.log_prob(batch_actions) * baseline_shift).mean()
            loss.backward()
            self.optimizer.step()
        

        if self.baseline_importance:
            baseline_shift = []
            batch_baseline = torch.tensor(np.full(len(batch_rewards), total_average_reward), requires_grad=False)
            baseline_shift = batch_rewards - batch_baseline

            for epoch in range(10):
                self.optimizer.zero_grad()
                logits = self.weights[batch_states]
                distribution = torch.distributions.categorical.Categorical(logits=logits)
                with torch.no_grad():
                    log_policy = distribution.log_prob(batch_actions)
                    ratio = torch.exp(log_policy - batch_log_policy)

                loss = - (self.env.max_num_steps * distribution.log_prob(batch_actions) * baseline_shift * ratio).mean()
                loss.backward()
                self.optimizer.step()


        if self.causality_importance:

            for epoch in range(10):
                self.optimizer.zero_grad()
                logits = self.weights[batch_states]
                distribution = torch.distributions.categorical.Categorical(logits=logits)
                with torch.no_grad():
                    log_policy = distribution.log_prob(batch_actions)
                    ratio = torch.exp(log_policy - batch_log_policy)

                loss = - (self.env.max_num_steps * distribution.log_prob(batch_actions) * batch_causality_rewards * ratio).mean()
                loss.backward()
                self.optimizer.step()

        
        if self.baseline_causality_importance:
            baseline_shift = []
            batch_baseline = torch.tensor(np.full(len(batch_causality_rewards), total_average_reward), requires_grad=False)
            baseline_shift = batch_causality_rewards - batch_baseline

            for epoch in range(10):
                self.optimizer.zero_grad()
                logits = self.weights[batch_states]
                distribution = torch.distributions.categorical.Categorical(logits=logits)
                with torch.no_grad():
                    log_policy = distribution.log_prob(batch_actions)
                    ratio = torch.exp(log_policy - batch_log_policy)

                loss = - (self.env.max_num_steps * distribution.log_prob(batch_actions) * baseline_shift * ratio).mean()
                loss.backward()
                self.optimizer.step()




    def save_model(self):

        if not os.path.exists('./model_save/' + str(self.model_name) + '/'):
            os.makedirs('./model_save/' + str(self.model_name) + '/')

        save_dir = './model_save/' + str(self.model_name) + '/' + str(self.episodes) + '_episodes.pt'

        torch.save({'log': self.log,
                    'weights': self.weights}, save_dir)

