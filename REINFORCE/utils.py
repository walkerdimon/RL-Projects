import numpy as np
import torch
import matplotlib.pyplot as plt
import pdb 

import os 
from os import listdir 
from os.path import isfile, join


def action_probability(self, s, a):

    #compute log-softmax probability of each action
    weights = self.weights.detach().numpy()
    softmax_prob = weights[s, a] - np.log(np.exp(weights[s]).sum())

    return softmax_prob



def action_probability_gradient(self, s, a):
    weights = self.weights.detach().numpy()
    probability = weights[s, a] - torch.exp(weights[s]).sum().log()
    probability.backward()

    probability_gradient = weights.grad.numpy()

    return probability_gradient



def get_action(self, s):

    probabilities = []

    #compute probabilites of each action
    for a in range(self.num_actions):
        prob = action_probability(self, s, a)
        probabilities.append(prob)

    #select random action based on computed probabilities
    action = np.random.choice(self.num_actions, p=probabilities)

    return action



def trajectory_probability(self, s, a, s1, a1):

    trajectory_probabilties = self.env.p0(s) * action_probabilities(s, a) * self.env.p(s1, s, a) * action_probabilities(s1, a1)

    return trajectory_probabilties



def trajectory_probability_gradient(self, s, a, s1, a1):
    
    probability_gradient = action_probability_gradient(s, a) + action_probability_gradient(s1, a1)

    return probability_gradient



def plot_learning_curve(args):

    reinforce_model = torch.load('./model_save/' + str(args.model_name) + '/' + str(args.episodes) + '_episodes.pt')
    model_log = reinforce_model['log']
    log_steps = model_log['steps']
    log_reward = model_log['reward']
    plt.plot(log_steps, log_reward)
    plt.xlabel('Simulation Steps')
    plt.ylabel('Total Reward')


    if not os.path.isdir('./generated_results/'):
        os.makedirs('./generated_results/')

    plt.savefig('./generated_results/' + str(args.model_name) + '_learning_curve_' + str(args.episodes) + '_Episodes.png' )


def generate_policy(args, env):

    reinforce_model = torch.load('./model_save/' + str(args.model_name) + '/' + str(args.episodes) + '_episodes.pt')
    weights = reinforce_model['weights']
    weights = weights.detach().numpy()
    policy = []
    for s in range(env.num_states):
        policy.append(np.argmax(weights[s]))

    policy = np.reshape(np.asarray(policy), (5, 5))

    grid_x = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    grid_y = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]


    fig = plt.figure()
    plt.imshow(policy, cmap='coolwarm')
    plt.colorbar()
    plt.xticks(grid_x)
    plt.yticks(grid_y)
    plt.grid(True)
    plt.title('REINFORCE Optimal Policy')
    fig.savefig('./generated_results/' + str(args.model_name) + '_policy_' + str(args.episodes) + '_Episodes.png' )
    plt.clf()

    pdb.set_trace()
