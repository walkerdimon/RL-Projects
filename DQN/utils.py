import random
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import torch
import torch.nn as nn
from collections import deque

import os
from os import listdir
from os.path import isfile, join

import discreteaction_pendulum
import pdb

from network import QNet



def e_greedy(self, s):

    if self.probability > self.epsilon:

        possible_a = []
        for a in range(self.num_actions):

            state = torch.tensor([s[0], s[1], a])
            a = self.QNet(state)
            possible_a.append(a.detach().numpy().copy())

        a = np.argmax(possible_a)

    else:
        a = random.randrange(self.num_actions)

    return a

def get_minibatch(self):
    memory_list = []
    for i in range(self.batch_size):
        memory_index = random.randint(0,len(self.memory)-1)
        memory_list.append(memory_index)

    minibatch = [self.memory[i] for i in memory_list]

    return minibatch


def get_y(self, done):

    r = []
    theta_list = []
    theta_dot_list = []
    for i in range(self.batch_size):
        r.append(self.minibatch[i][3])
        theta_list.append(self.minibatch[i][4])
        theta_dot_list.append(self.minibatch[i][5])

    if done == True:
        y = torch.tensor([r], dtype=torch.float32)

    else:
        a_list = []
        for i in range(self.batch_size):

            possible_a = []
            for a in range(self.num_actions):

                state = torch.tensor([theta_list[i], theta_dot_list[i], a])
                action = self.QNet(state)
                possible_a.append(action.detach().numpy().copy())

            a_list.append(np.argmax(possible_a))   

        states = torch.transpose(torch.tensor([theta_list, theta_dot_list, a_list]), 0, 1)

        Q = self.Q_hat(states)

        y = torch.transpose(torch.tensor([r], dtype=torch.float32), 0, 1) + (self.gamma * Q)

    return y

def get_Q(self):

    theta = []
    theta_dot = []
    a_list = []

    for i in range(self.batch_size):
        theta.append(self.minibatch[i][0])
        theta_dot.append(self.minibatch[i][1])
        a_list.append(self.minibatch[i][2])

    states = torch.transpose(torch.tensor([theta, theta_dot, a_list]), 0, 1)
    Q = self.QNet(states)

    return Q



def plot_learning_curve(self):

    log = np.mean(self.model_log['r'].reshape(-1,1000), axis=1)
    plt.plot(log, label='r')
    plt.xlabel('Time Interval')
    plt.ylabel('Discounted Reward')
    plt.legend()
    plt.savefig('./generated_results/Learning_Curve_' + str(self.max_episodes) + '_Episodes.png' )



def plot_example_trajectory(self):
    s = self.env.reset()

    # Create dict to store data from simulation
    data = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }

    # Simulate until episode is done
    done = False
    while not done:
        possible_a = []
        for a in range(self.num_actions):
            state = torch.tensor([s[0], s[1], a])
            Q = self.QNet(state)
            possible_a.append(Q.detach().numpy().copy())

        a = np.argmax(possible_a)
        (s, r, done) = self.env.step(a)
        data['t'].append(data['t'][-1] + 1)
        data['s'].append(s)
        data['a'].append(a)
        data['r'].append(r)

    # Parse data from simulation
    data['s'] = np.array(data['s'])
    theta = data['s'][:, 0]
    thetadot = data['s'][:, 1]
    tau = [self.env._a_to_u(a) for a in data['a']]

    # Plot data and save to png file
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(data['t'], theta, label='theta')
    ax[0].plot(data['t'], thetadot, label='thetadot')
    ax[0].legend()
    ax[1].plot(data['t'][:-1], tau, label='tau')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig('./generated_results/Example_Trajectory_' + str(self.max_episodes) + '_Episodes.png' )



def generate_animated_trajectory(self, filename='pendulum.gif', writer='imagemagick'):
        s = self.env.reset()
        s_traj = [s]
        done = False
        while not done:
            possible_a = []
            for a in range(self.num_actions):
                state = torch.tensor([s[0], s[1], a])
                Q = self.QNet(state)
                possible_a.append(Q.detach().numpy().copy())

            a = np.argmax(possible_a)
            (s, r, done) = self.env.step(a)
            s_traj.append(s)

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
        ax.set_aspect('equal')
        ax.grid()
        line, = ax.plot([], [], 'o-', lw=2)
        text = ax.set_title('')

        def animate(i):
            theta = s_traj[i][0]
            line.set_data([0, -np.sin(theta)], [0, np.cos(theta)])
            text.set_text(f'time = {i * self.env.dt:3.1f}')
            return line, text

        anim = animation.FuncAnimation(fig, animate, len(s_traj), interval=(1000 * self.env.dt), blit=True, repeat=False)
        anim.save(filename, writer=writer, fps=10)

        plt.close()

def value_visualization(self):
    theta_range = np.linspace(-np.pi, np.pi, 100)
    theta_dot_range = np.linspace(-self.env.max_thetadot_for_init, self.env.max_thetadot_for_init, 100)

    Q_list = []
    for i in range(len(theta_range)):
        Q_row_list = []
        for j in range(len(theta_dot_range)):

            possible_Q = []
            for a in range(self.num_actions):
                state = torch.tensor([theta_range[i], theta_dot_range[j], a])
                Q = self.QNet(state)
                possible_Q.append(Q.detach().numpy().copy())

            Q = np.max(possible_Q)

            Q_row_list.append(Q)
        
        Q_list.append(Q_row_list)

    fig = plt.figure()
    plt.imshow(Q_list, cmap='coolwarm')
    plt.xlabel('theta dot')
    plt.ylabel('theta')
    plt.colorbar()
    plt.title('Value Function Visualization')
    fig.savefig('./generated_results/Value_Visualization_' + str(self.max_episodes) + '_Episodes.png')
    fig.clf()

          
def policy_visualization(self):
    theta_range = np.linspace(-np.pi, np.pi, 100)
    theta_dot_range = np.linspace(-self.env.max_thetadot_for_init, self.env.max_thetadot_for_init, 100)

    policy_list = []
    for i in range(len(theta_range)):
        policy_row_list = []
        for j in range(len(theta_dot_range)):

            possible_a = []
            for a in range(self.num_actions):
                state = torch.tensor([theta_range[i], theta_dot_range[j], a])
                Q = self.QNet(state)
                possible_a.append(Q.detach().numpy().copy())

            policy = self.env._a_to_u(np.argmax(possible_a))

            policy_row_list.append(policy)
        
        policy_list.append(policy_row_list)

    fig = plt.figure()
    plt.imshow(policy_list, cmap='coolwarm')
    plt.xlabel('theta dot')
    plt.ylabel('theta')
    plt.colorbar()
    plt.title('Policy Visualization')
    fig.savefig('./generated_results/Policy_Visualization_' + str(self.max_episodes) + '_Episodes.png')
    fig.clf()



