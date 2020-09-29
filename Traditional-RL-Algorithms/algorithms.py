import random
import numpy as np
import matplotlib.pyplot as plt
import argparse

import gridworld
import discrete_pendulum
import pdb



def ValueIteration(args, P, R, V, policy, env):

    optimal = False
    mean_VF_list = [] #store mean value function for plotting learning curves
    iterations_list = [] #store iterations for plotting learning curves
    iterations = 0

    while optimal == False:

        mean_VF_list.append(np.mean(V))
        iterations += 1
        iterations_list.append(iterations)

        delta = 0

        for s in range(env.num_states):

            v = V[s]

            V_sample = []
            for a in range(env.num_actions):
                V_sample.append(sum([(P(s1, s, a) * (args.gamma * V[s1] + R(s, a))) for s1 in range(env.num_states)]))

            V[s] = max(V_sample)

            delta = max(delta, abs(v - V[s]))

            if delta < args.accuracy_limit:
                optimal = True
            else:
                optimal = False


        if optimal == True:

            for s in range(env.num_states):

                V_sample = []
                for a in range(env.num_actions):
                    V_sample.append(sum([P(s1, s, a) * (R(s, a) + (args.gamma * V[s1])) for s1 in range(env.num_states)]))

                policy[s] = np.argmax(V_sample)

                                
    return (V, policy, mean_VF_list, iterations_list)             


def PolicyIteration(args, P, R, V, policy, env):

    optimal = False
    mean_VF_list = [] #store mean value function for plotting learning curves
    iterations_list = [] #store iterations for plotting learning curves
    iterations = 0

    while optimal == False:
        mean_VF_list.append(np.mean(V))
        iterations += 1
        iterations_list.append(iterations)

        delta = 0

        for s in range(env.num_states):

            v = V[s]

            V[s] = sum([P(s1, s, policy[s]) * (R(s,policy[s]) + (args.gamma * V[s1])) for s1 in range(env.num_states)])

            delta = max(delta, abs(v - V[s]))

            if delta < args.accuracy_limit:
                optimal = True

            else:
                optimal = False


        if optimal == True:



            for s in range(env.num_states):

                old_action = policy[s]

                policy[s] = np.argmax([sum([P(s1,s,a) * (R(s,a) + (args.gamma * V[s1])) for s1 in range(env.num_states)]) for a in range(env.num_actions)])

                if policy[s] != old_action:
                    optimal = False

    return(V, policy, mean_VF_list, iterations_list)




def SARSA(args, Q, env):
    probability = random.uniform(0,1)

    discounted_reward = []

    def e_greedy(args, Q, s, env):
    
        if probability < args.epsilon:
            a = np.argmax(Q[s][:])

        else:
            a = random.randrange(env.num_actions)

        return a


    for episode in range(args.max_episodes):
        print(episode)
        reward = 0

        #initialize state and action
        s = env.reset()
        a = e_greedy(args, Q, s, env)

        done = False

        while done == False:

            s1, r, done = env.step(a)
            reward += r
            a1 = e_greedy(args, Q, s1, env)
            Q[s][a] = Q[s][a] + (args.alpha * (r + (args.gamma * Q[s1][a1]) - Q[s][a]))

            s = s1
            a = a1
        discounted_reward.append(reward)

    policy = []
    for s in range(env.num_states):

        policy.append(np.argmax(Q[s]))

    return Q, policy, discounted_reward




def Q_learning(args, Q, env):

    probability = random.uniform(0,1)

    discounted_reward = []

    def e_greedy(args, Q, s, env):
    
        if probability < args.epsilon:
            a = np.argmax([Q[s][x] for x in range(env.num_actions)])
            

        else:
            a = random.randrange(env.num_actions)

        return a

    for episode in range(args.max_episodes):
        print(episode)
        reward = 0

        #initialize state and action
        s = env.reset()

        done = False

        while done == False:

            a = e_greedy(args, Q, s, env)

            s1, r, done = env.step(a)
            reward += r

            a1 = e_greedy(args, Q, s1, env)
            Q[s][a] = Q[s][a] + (args.alpha * (r + (args.gamma * Q[s1][a1]) - Q[s][a]))

            s = s1
        discounted_reward.append(reward)


    policy = []
    for s in range(env.num_states):

        policy.append(np.argmax(Q[s]))

    return Q, policy, discounted_reward



def TD_Zero(args, V, policy, env):


    for episode in range(args.max_episodes):
        print(episode)

        #intialize s
        s = env.reset()

        done = False
        
        while done == False:

            s1, r, done = env.step(policy[s])

            V[s] = V[s] + (args.alpha * (r + (args.gamma*V[s1]) - V[s]) )

            s = s1


    return V

        



