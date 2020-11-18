import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt 
from matplotlib.font_manager import FontProperties

import gridworld

import os 
from os import listdir 
from os.path import isfile, join
import pickle
import pdb

base1 = torch.load('./model_save/base1/10000_episodes.pt')
base1_log = base1['log']
base1_weights = base1['weights']
base1_rewards = np.mean(np.array(base1_log['reward']).reshape(-1,10), axis=1)

base2 = torch.load('./model_save/base2/10000_episodes.pt')
base2_log = base2['log']
base2_weights = base2['weights']
base2_rewards = np.mean(np.array(base2_log['reward']).reshape(-1,10), axis=1)

baseline_importance1 = torch.load('./model_save/baseline_importance1/10000_episodes.pt')
baseline_importance1_log = baseline_importance1['log']
baseline_importance1_rewards = np.mean(np.array(baseline_importance1_log['reward']).reshape(-1,10), axis=1)

baseline_importance2 = torch.load('./model_save/baseline_importance2/10000_episodes.pt')
baseline_importance2_log = baseline_importance2['log']
baseline_importance2_rewards = np.mean(np.array(baseline_importance2_log['reward']).reshape(-1,10), axis=1)

baseline_causality1 = torch.load('./model_save/baseline_causality1/10000_episodes.pt')
baseline_causality1_log = baseline_causality1['log']
baseline_causality1_rewards = np.mean(np.array(baseline_causality1_log['reward']).reshape(-1,10), axis=1)

baseline_causality2 = torch.load('./model_save/baseline_causality2/10000_episodes.pt')
baseline_causality2_log = baseline_causality2['log']
baseline_causality2_rewards = np.mean(np.array(baseline_causality2_log['reward']).reshape(-1,10), axis=1)

causality_importance1 = torch.load('./model_save/causality_importance1/10000_episodes.pt')
causality_importance1_log = causality_importance1['log']
causality_importance1_rewards = np.mean(np.array(causality_importance1_log['reward']).reshape(-1,10), axis=1)

causality_importance2 = torch.load('./model_save/causality_importance2/10000_episodes.pt')
causality_importance2_log = causality_importance2['log']
causality_importance2_rewards = np.mean(np.array(causality_importance2_log['reward']).reshape(-1,10), axis=1)

baseline_causality_importance1 = torch.load('./model_save/baseline_causality_importance1/10000_episodes.pt')
baseline_causality_importance1_log = baseline_causality_importance1['log']
baseline_causality_importance1_rewards = np.mean(np.array(baseline_causality_importance1_log['reward']).reshape(-1,10), axis=1)

baseline_causality_importance2 = torch.load('./model_save/baseline_causality_importance2/10000_episodes.pt')
baseline_causality_importance2_log = baseline_causality_importance2['log']
baseline_causality_importance2_rewards = np.mean(np.array(baseline_causality_importance2_log['reward']).reshape(-1,10), axis=1)


ADAM_base1 = torch.load('./model_save/ADAM_base1/10000_episodes.pt')
ADAM_base1_log = ADAM_base1['log']
ADAM_base1_weights = ADAM_base1['weights']
ADAM_base1_rewards = np.mean(np.array(ADAM_base1_log['reward']).reshape(-1,10), axis=1)

ADAM_base2 = torch.load('./model_save/ADAM_base2/10000_episodes.pt')
ADAM_base2_log = ADAM_base2['log']
ADAM_base2_weights = ADAM_base2['weights']
ADAM_base2_rewards = np.mean(np.array(ADAM_base2_log['reward']).reshape(-1,10), axis=1)

ADAM_baseline_importance1 = torch.load('./model_save/ADAM_baseline_importance1/10000_episodes.pt')
ADAM_baseline_importance1_log = ADAM_baseline_importance1['log']
ADAM_baseline_importance1_rewards = np.mean(np.array(ADAM_baseline_importance1_log['reward']).reshape(-1,10), axis=1)

ADAM_baseline_importance2 = torch.load('./model_save/ADAM_baseline_importance2/10000_episodes.pt')
ADAM_baseline_importance2_log = ADAM_baseline_importance2['log']
ADAM_baseline_importance2_rewards = np.mean(np.array(ADAM_baseline_importance2_log['reward']).reshape(-1,10), axis=1)

ADAM_baseline_causality1 = torch.load('./model_save/ADAM_baseline_causality1/10000_episodes.pt')
ADAM_baseline_causality1_log = ADAM_baseline_causality1['log']
ADAM_baseline_causality1_rewards = np.mean(np.array(ADAM_baseline_causality1_log['reward']).reshape(-1,10), axis=1)

ADAM_baseline_causality2 = torch.load('./model_save/ADAM_baseline_causality2/10000_episodes.pt')
ADAM_baseline_causality2_log = ADAM_baseline_causality2['log']
ADAM_baseline_causality2_rewards = np.mean(np.array(ADAM_baseline_causality2_log['reward']).reshape(-1,10), axis=1)

ADAM_causality_importance1 = torch.load('./model_save/ADAM_causality_importance1/10000_episodes.pt')
ADAM_causality_importance1_log = ADAM_causality_importance1['log']
ADAM_causality_importance1_rewards = np.mean(np.array(ADAM_causality_importance1_log['reward']).reshape(-1,10), axis=1)

ADAM_causality_importance2 = torch.load('./model_save/ADAM_causality_importance2/10000_episodes.pt')
ADAM_causality_importance2_log = ADAM_causality_importance2['log']
ADAM_causality_importance2_rewards = np.mean(np.array(ADAM_causality_importance2_log['reward']).reshape(-1,10), axis=1)

ADAM_baseline_causality_importance1 = torch.load('./model_save/ADAM_baseline_causality_importance1/10000_episodes.pt')
ADAM_baseline_causality_importance1_log = ADAM_baseline_causality_importance1['log']
ADAM_baseline_causality_importance1_rewards = np.mean(np.array(ADAM_baseline_causality_importance1_log['reward']).reshape(-1,10), axis=1)

ADAM_baseline_causality_importance2 = torch.load('./model_save/ADAM_baseline_causality_importance2/10000_episodes.pt')
ADAM_baseline_causality_importance2_log = ADAM_baseline_causality_importance2['log']
ADAM_baseline_causality_importance2_rewards = np.mean(np.array(ADAM_baseline_causality_importance2_log['reward']).reshape(-1,10), axis=1)


avg_base = (base1_rewards + base2_rewards) / 2.0

avg_baseline_causality = (baseline_causality1_rewards + baseline_causality2_rewards) / 2.0

avg_baseline_importance = (baseline_importance1_rewards + baseline_importance2_rewards) / 2.0

avg_causality_importance = (causality_importance1_rewards + causality_importance2_rewards) / 2.0

avg_baseline_causality_importance = (baseline_causality_importance1_rewards + baseline_causality_importance2_rewards) / 2.0



ADAM_avg_base = (ADAM_base1_rewards + ADAM_base2_rewards) / 2.0

ADAM_avg_baseline_causality = (ADAM_baseline_causality1_rewards + ADAM_baseline_causality2_rewards) / 2.0

ADAM_avg_baseline_importance = (ADAM_baseline_importance1_rewards + ADAM_baseline_importance2_rewards) / 2.0

ADAM_avg_causality_importance = (ADAM_causality_importance1_rewards + ADAM_causality_importance2_rewards) / 2.0

ADAM_avg_baseline_causality_importance = (ADAM_baseline_causality_importance1_rewards + ADAM_baseline_causality_importance2_rewards) / 2.0



if not os.path.isdir('./generated_results/'):
    os.makedirs('./generated_results/')

fig = plt.figure()
plt.plot(avg_base, label='Base Model')
plt.xlabel('Simulation Steps (10 Episodes/Step)')
plt.ylabel('Total Reward')
plt.title('Base Model Learning Curve: 10,000 Episodes')
plt.legend()
plt.savefig('./generated_results/base_learning_curve_10000_episodes.png' )




fig = plt.figure()
plt.plot(avg_base, label='Base Model')
plt.plot(avg_baseline_importance, label='BS + IS')
plt.plot(avg_baseline_causality, label='BS + C')
plt.plot(avg_baseline_causality_importance, label='BS + C + IS')
plt.plot(avg_causality_importance, label='C + IS')
plt.xlabel('Simulation Steps (10 Episodes/Step)')
plt.ylabel('Total Reward')
plt.title('Learning Curve Comparison: 10,000 Episodes')
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.savefig('./generated_results/comparison_learning_curve_10000_episodes.png' )
plt.show()


fig = plt.figure()
plt.plot(avg_base, label='Base Model')
plt.plot(avg_baseline_importance, label='BS + IS')
plt.plot(avg_baseline_causality, label='BS + C')
plt.plot(avg_baseline_causality_importance, label='BS + C + IS')
plt.plot(avg_causality_importance, label='C + IS')
plt.plot(ADAM_avg_base, '--', label='ADAM: Base Model')
plt.plot(ADAM_avg_baseline_importance, '--', label='ADAM: BS + IS')
plt.plot(ADAM_avg_baseline_causality, '--', label='ADAM: BS + C')
plt.plot(ADAM_avg_baseline_causality_importance, '--', label='ADAM: BS + C + IS')
plt.plot(ADAM_avg_causality_importance, '', label='ADAM: C+ IS')
plt.xlabel('Simulation Steps (10 Episodes/Step)')
plt.ylabel('Total Reward')
plt.title('Learning Curve Comparison (SGD vs ADAM): 10,000 Episodes')
plt.legend(bbox_to_anchor=(.975, 1.0), loc='upper left')
plt.savefig('./generated_results/SGD_ADAM_learning_curve_10000_episodes.png' )
plt.show()


env = gridworld.GridWorld()

weights = base1_weights.detach().numpy()
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
plt.title('REINFORCE Optimal Policy: 10,000 Episodes (SGD Optimizer)')
fig.savefig('./generated_results/SGD_Base_Policy_10000_Episodes.png' )
plt.clf()

ADAM_weights = ADAM_base1_weights.detach().numpy()
policy = []
for s in range(env.num_states):
    policy.append(np.argmax(ADAM_weights[s]))

policy = np.reshape(np.asarray(policy), (5, 5))

grid_x = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
grid_y = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]


fig = plt.figure()
plt.imshow(policy, cmap='coolwarm')
plt.colorbar()
plt.xticks(grid_x)
plt.yticks(grid_y)
plt.grid(True)
plt.title('REINFORCE Optimal Policy: 10,000 Episodes (Adam Optimizer)')
fig.savefig('./generated_results/ADAM_Base_Policy_10000_Episodes.png' )
plt.clf()