import random
import numpy as np
import matplotlib.pyplot as plt
import argparse

import gridworld
import discrete_pendulum
from algorithms import ValueIteration, PolicyIteration, SARSA, Q_learning, TD_Zero

import os
from os import listdir
from os.path import isfile, join
import pickle

import pdb


#Input Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--pendulum", action='store_true')
parser.add_argument("--gridworld", action='store_true')
parser.add_argument("--value_iteration", action='store_true')
parser.add_argument("--policy_iteration", action='store_true')
parser.add_argument("--SARSA", action='store_true')
parser.add_argument("--q_learning", action='store_true')
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--epsilon", type=float, default=0.1)
parser.add_argument("--accuracy_limit", type=float, default=0.01)
parser.add_argument("--max_episodes", type=int, default=200)

args = parser.parse_args()



def main():

    #Argument to initialize grid world
    if args.gridworld:

        #create environment
        env = gridworld.GridWorld(hard_version=False)

        #initializations
        P = env.p # state transition probability
        R = env.r # reward function
        V = np.zeros(env.num_states) #state value function
        policy = np.zeros(env.num_states)
        Q = np.zeros((env.num_states, env.num_actions)) #state action value function

        #Argument to run Value Iteration
        if args.value_iteration:

            V_optimal, policy_optimal, mean_VF_list, iterations = ValueIteration(args, P, R, V, policy, env)

            data = [V_optimal, policy_optimal, mean_VF_list, iterations]


            #save data to pickle for deliverable generation
            if not os.path.isdir('./gridworld_data/'):
                os.makedirs('./gridworld_data/')
            filename = 'value_iteration_data.pkl'
            with open('gridworld_data/' + filename, 'wb') as f:
                pickle.dump(data, f)


        #Argument to run Policy Iteration
        if args.policy_iteration:

            V_optimal, policy_optimal, mean_VF_list, iterations = PolicyIteration(args, P, R, V, policy, env)

            data = [V_optimal, policy_optimal, mean_VF_list, iterations]

            #save data to pickle for deliverable generation
            if not os.path.isdir('./gridworld_data/'):
                os.makedirs('./gridworld_data/')
            filename = 'policy_iteration_data.pkl'
            with open('gridworld_data/' + filename, 'wb') as f:
                pickle.dump(data, f)



        #Argument to run SARSA
        if args.SARSA:

            Q_optimal, policy_optimal, discounted_rewards = SARSA(args, Q, env)

            V_estimate = TD_Zero(args, V, policy_optimal, env)


            data = [Q_optimal, policy_optimal, discounted_rewards, V_estimate]


            #save data to pickle for deliverable generation
            if not os.path.isdir('./gridworld_data/'):
                os.makedirs('./gridworld_data/')
            filename = 'SARSA_data_alpha=' + str(args.alpha) + '_epsilon=' + str(args.epsilon) + '.pkl'
            with open('gridworld_data/' + filename, 'wb') as f:
                pickle.dump(data, f)



        #Argument to run Q learning
        if args.q_learning:

            Q_optimal, policy_optimal, discounted_rewards = Q_learning(args, Q, env)

            V_estimate = TD_Zero(args, V, policy_optimal, env)

            data = [Q_optimal, policy_optimal, discounted_rewards, V_estimate]

            #save data to pickle for deliverable generation
            if not os.path.isdir('./gridworld_data/'):
                os.makedirs('./gridworld_data/')
            filename = 'Q_Learning_data_alpha=' + str(args.alpha) + '_epsilon=' + str(args.epsilon) + '.pkl'
            with open('gridworld_data/' + filename, 'wb') as f:
                pickle.dump(data, f)





    #Argument to initialize grid world
    if args.pendulum:

        env = discrete_pendulum.Pendulum()

        #Initializations
        Q = np.zeros((env.num_states, env.num_actions))
        V = np.zeros(env.num_states) #state value function

        #Argument to run SARSA
        if args.SARSA:

            Q_optimal, policy_optimal, discounted_rewards = SARSA(args, Q, env)

            V_estimate = TD_Zero(args, V, policy_optimal, env)

            data = [Q_optimal, policy_optimal, discounted_rewards, V_estimate]


            #save data to pickle for deliverable generation
            if not os.path.isdir('./pendulum_data/'):
                os.makedirs('./pendulum_data/')
            filename = 'SARSA_data_alpha=' + str(args.alpha) + '_epsilon=' + str(args.epsilon) + '.pkl'
            with open('pendulum_data/' + filename, 'wb') as f:
                pickle.dump(data, f)


        #Argument to run Q learning    
        if args.q_learning:

            Q_optimal, policy_optimal, discounted_rewards = Q_learning(args, Q, env)

            V_estimate = TD_Zero(args, V, policy_optimal, env)

            data = [Q_optimal, policy_optimal, discounted_rewards, V_estimate]
               
            #save data to pickle for deliverable generation
            if not os.path.isdir('./pendulum_data/'):
                os.makedirs('./pendulum_data/')
            filename = 'Q_Learning_data_alpha=' + str(args.alpha) + '_epsilon=' + str(args.epsilon) + '.pkl'
            with open('pendulum_data/' + filename, 'wb') as f:
                pickle.dump(data, f)



if __name__ == '__main__':
    main()
