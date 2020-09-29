import matplotlib.pyplot as plt
import numpy as np


import gridworld
import discrete_pendulum


import os
from os import listdir
from os.path import isfile, join
import pickle

import pdb

with open('./gridworld_data/policy_iteration_data.pkl', 'rb') as f:
    policy_iteration_data = pickle.load(f) #[V_optimal, policy_optimal, mean_VF_list, iterations]

with open('./gridworld_data/value_iteration_data.pkl', 'rb') as f:
    value_iteration_data = pickle.load(f) #[V_optimal, policy_optimal, mean_VF_list, iterations]

with open('./gridworld_data/Q_Learning_data_alpha=0.4_epsilon=0.1.pkl', 'rb') as f:
    grid_Q_Learning_data_4_1 = pickle.load(f) #[Q_optimal, policy_optimal, discounted_rewards, V_estimate]

with open('./gridworld_data/Q_Learning_data_alpha=0.5_epsilon=0.1.pkl', 'rb') as f:
    grid_Q_Learning_data_5_1 = pickle.load(f) #[Q_optimal, policy_optimal, discounted_rewards, V_estimate]

with open('./gridworld_data/Q_Learning_data_alpha=0.5_epsilon=0.2.pkl', 'rb') as f:
    grid_Q_Learning_data_5_2 = pickle.load(f) #[Q_optimal, policy_optimal, discounted_rewards, V_estimate]

with open('./gridworld_data/Q_Learning_data_alpha=0.5_epsilon=0.3.pkl', 'rb') as f:
    grid_Q_Learning_data_5_3 = pickle.load(f) #[Q_optimal, policy_optimal, discounted_rewards, V_estimate]

with open('./gridworld_data/Q_Learning_data_alpha=0.6_epsilon=0.1.pkl', 'rb') as f:
    grid_Q_Learning_data_6_1 = pickle.load(f) #[Q_optimal, policy_optimal, discounted_rewards, V_estimate]

with open('./gridworld_data/SARSA_data_alpha=0.4_epsilon=0.1.pkl', 'rb') as f:
    grid_SARSA_data_4_1 = pickle.load(f) #[Q_optimal, policy_optimal, discounted_rewards, V_estimate]

with open('./gridworld_data/SARSA_data_alpha=0.5_epsilon=0.1.pkl', 'rb') as f:
    grid_SARSA_data_5_1 = pickle.load(f) #[Q_optimal, policy_optimal, discounted_rewards, V_estimate]

with open('./gridworld_data/SARSA_data_alpha=0.5_epsilon=0.2.pkl', 'rb') as f:
    grid_SARSA_data_5_2 = pickle.load(f) #[Q_optimal, policy_optimal, discounted_rewards, V_estimate]

with open('./gridworld_data/SARSA_data_alpha=0.5_epsilon=0.3.pkl', 'rb') as f:
    grid_SARSA_data_5_3 = pickle.load(f) #[Q_optimal, policy_optimal, discounted_rewards, V_estimate]

with open('./gridworld_data/SARSA_data_alpha=0.6_epsilon=0.1.pkl', 'rb') as f:
    grid_SARSA_data_6_1 = pickle.load(f) #[Q_optimal, policy_optimal, discounted_rewards, V_estimate]



with open('./pendulum_data/Q_Learning_data_alpha=0.4_epsilon=0.1.pkl', 'rb') as f:
    pendulum_Q_Learning_data_4_1 = pickle.load(f) #[Q_optimal, policy_optimal, discounted_rewards, V_estimate]

with open('./pendulum_data/Q_Learning_data_alpha=0.5_epsilon=0.1.pkl', 'rb') as f:
    pendulum_Q_Learning_data_5_1 = pickle.load(f) #[Q_optimal, policy_optimal, discounted_rewards, V_estimate]

with open('./pendulum_data/Q_Learning_data_alpha=0.5_epsilon=0.2.pkl', 'rb') as f:
    pendulum_Q_Learning_data_5_2 = pickle.load(f) #[Q_optimal, policy_optimal, discounted_rewards, V_estimate]

with open('./pendulum_data/Q_Learning_data_alpha=0.5_epsilon=0.3.pkl', 'rb') as f:
    pendulum_Q_Learning_data_5_3 = pickle.load(f) #[Q_optimal, policy_optimal, discounted_rewards, V_estimate]

with open('./pendulum_data/Q_Learning_data_alpha=0.6_epsilon=0.1.pkl', 'rb') as f:
    pendulum_Q_Learning_data_6_1 = pickle.load(f) #[Q_optimal, policy_optimal, discounted_rewards, V_estimate]

with open('./pendulum_data/SARSA_data_alpha=0.4_epsilon=0.1.pkl', 'rb') as f:
    pendulum_SARSA_data_4_1 = pickle.load(f) #[Q_optimal, policy_optimal, discounted_rewards, V_estimate]

with open('./pendulum_data/SARSA_data_alpha=0.5_epsilon=0.1.pkl', 'rb') as f:
    pendulum_SARSA_data_5_1 = pickle.load(f) #[Q_optimal, policy_optimal, discounted_rewards, V_estimate]

with open('./pendulum_data/SARSA_data_alpha=0.5_epsilon=0.2.pkl', 'rb') as f:
    pendulum_SARSA_data_5_2 = pickle.load(f) #[Q_optimal, policy_optimal, discounted_rewards, V_estimate]

with open('./pendulum_data/SARSA_data_alpha=0.5_epsilon=0.3.pkl', 'rb') as f:
    pendulum_SARSA_data_5_3 = pickle.load(f) #[Q_optimal, policy_optimal, discounted_rewards, V_estimate]

with open('./pendulum_data/SARSA_data_alpha=0.6_epsilon=0.1.pkl', 'rb') as f:
    pendulum_SARSA_data_6_1 = pickle.load(f) #[Q_optimal, policy_optimal, discounted_rewards, V_estimate]







#POLICY PLOTS:
if not os.path.isdir('./policy_visualization/'):
    os.makedirs('./policy_visualization/')


grid_x = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
grid_y = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]

pend_x = [(0.5 + x) for x in range(31)]
pend_y = pend_x


#Value Function
v_policy = np.reshape(value_iteration_data[1], (5, 5))
fig = plt.figure()
plt.imshow(v_policy, cmap='coolwarm')
plt.colorbar()
plt.xticks(grid_x)
plt.yticks(grid_y)
plt.grid(True)
plt.title('Value Iteration Optimal Policy (Gridworld)')
fig.savefig('./policy_visualization/Value Iteration.png')
plt.clf()


#Policy Function
p_policy = np.reshape(policy_iteration_data[1], (5, 5))
fig = plt.figure()
plt.imshow(p_policy, cmap='coolwarm')
plt.colorbar()
plt.xticks(grid_x)
plt.yticks(grid_y)
plt.grid(True)
plt.title('Policy Iteration Optimal Policy (Gridworld)')
fig.savefig('./policy_visualization/Policy Iteration.png')
plt.clf()

#SARSA (grid)
sg_policy = np.reshape(grid_SARSA_data_5_1[1], (5, 5))
fig = plt.figure()
plt.imshow(sg_policy, cmap='coolwarm')
plt.colorbar()
plt.xticks(grid_x)
plt.yticks(grid_y)
plt.grid(True)
plt.title('SARSA Optimal Policy (Gridworld)')
fig.savefig('./policy_visualization/SARSA (gridworld).png')
plt.clf()

#Q-learning (grid)
qg_policy = np.reshape(grid_Q_Learning_data_5_1[1], (5, 5))
fig = plt.figure()
plt.imshow(qg_policy, cmap='coolwarm')
plt.colorbar()
plt.xticks(grid_x)
plt.yticks(grid_y)
plt.grid(True)
plt.title('Q Learning Optimal Policy (Gridworld)')
fig.savefig('./policy_visualization/Q Learning (gridworld).png')
plt.clf()

#SARSA (pendulum)
sp_policy = np.reshape(pendulum_SARSA_data_5_1[1], (31, 31))
fig = plt.figure()
plt.imshow(sp_policy, cmap='coolwarm')
plt.colorbar()
plt.xticks(pend_x)
plt.yticks(pend_y)
plt.grid(True)
plt.title('SARSA Optimal Policy (Pendulum)')
fig.savefig('./policy_visualization/SARSA (pendulum).png')
plt.clf()

#Q-learning (pendulum)
qp_policy = np.reshape(pendulum_Q_Learning_data_5_1[1], (31, 31))
fig = plt.figure()
plt.imshow(qp_policy, cmap='coolwarm')
plt.colorbar()
plt.xticks(pend_x)
plt.yticks(pend_y)
plt.grid(True)
plt.title('Q Learning Optimal Policy (Pendulum)')
fig.savefig('./policy_visualization/Q Learning (pendulum).png')
plt.clf()






#VALUE FUNCTION PLOTS
if not os.path.isdir('./value_function_visualization/'):
    os.makedirs('./value_function_visualization/')


#SARSA (pendulum)
sp_policy = np.reshape(pendulum_SARSA_data_5_1[3], (31, 31))
fig = plt.figure()
plt.imshow(sp_policy, cmap='coolwarm')
plt.colorbar()
plt.xticks(pend_x)
plt.yticks(pend_y)
plt.grid(True)
plt.title('SARSA Value Function (Pendulum)')
fig.savefig('./value_function_visualization/SARSA (pendulum).png')
plt.clf()

#Q-learning (pendulum)
qp_policy = np.reshape(pendulum_Q_Learning_data_5_1[3], (31, 31))
fig = plt.figure()
plt.imshow(qp_policy, cmap='coolwarm')
plt.colorbar()
plt.xticks(pend_x)
plt.yticks(pend_y)
plt.grid(True)
plt.title('Q Learning Value Function (Pendulum)')
fig.savefig('./value_function_visualization/Q Learning (pendulum).png')
plt.clf()

pdb.set_trace()



#SAMPLE TRAJECTORIES
if not os.path.isdir('./example_trajectories/'):
    os.makedirs('./example_trajectories/')

env = discrete_pendulum.Pendulum()

# Initialize simulation
s = env.reset()

# Create log to store data from simulation
log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
        'theta': [env.x[0]],        # agent does not have access to this, but helpful for display
        'thetadot': [env.x[1]],     # agent does not have access to this, but helpful for display
    }

# Simulate until episode is done
done = False
while not done:
    policy = pendulum_SARSA_data_5_2[1]
    (s, r, done) = env.step(policy[s])
    log['t'].append(log['t'][-1] + 1)
    log['s'].append(s)
    log['a'].append(policy[s])
    log['r'].append(r)
    log['theta'].append(env.x[0])
    log['thetadot'].append(env.x[1])

# Plot data and save to png file
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
fig.suptitle('Example Pendulum Trajectory: SARSA', fontsize=16)
ax[0].plot(log['t'], log['s'])
ax[0].plot(log['t'][:-1], log['a'])
ax[0].plot(log['t'][:-1], log['r'])
ax[0].legend(['s', 'a', 'r'])
ax[1].plot(log['t'], log['theta'])
ax[1].plot(log['t'], log['thetadot'])
ax[1].legend(['theta', 'thetadot'])
plt.savefig('./example_trajectories/SARSA_discrete_pendulum.png')


pdb.set_trace()




#ALPHA-EPSILON SWEEP(SARSA)
if not os.path.isdir('./SARSA_sweep/gridworld'):
    os.makedirs('./SARSA_sweep/gridworld')

#alpha = 0.4, epsilon = 0.1
fig = plt.figure()
plt.plot(range(len(grid_SARSA_data_4_1[2])), grid_SARSA_data_4_1[2])
plt.xlabel('Iterations')
plt.ylabel('Total Discounted Rewards')
plt.title('SARSA Learning Curve: alpha=0.4 epsilon=0.1 (Gridworld)')
fig.savefig('./SARSA_sweep/gridworld/alpha0.4_epsilon0.1.png')
plt.clf()

#alpha = 0.5, epsilon = 0.1
fig = plt.figure()
plt.plot(range(len(grid_SARSA_data_5_1[2])), grid_SARSA_data_5_1[2])
plt.xlabel('Iterations')
plt.ylabel('Total Discounted Rewards')
plt.title('SARSA Learning Curve: alpha=0.5 epsilon=0.1 (Gridworld)')
fig.savefig('./SARSA_sweep/gridworld/alpha0.5_epsilon0.1.png')
plt.clf()

#alpha = 0.5, epsilon = 0.2
fig = plt.figure()
plt.plot(range(len(grid_SARSA_data_5_2[2])), grid_SARSA_data_5_2[2])
plt.xlabel('Iterations')
plt.ylabel('Total Discounted Rewards')
plt.title('SARSA Learning Curve: alpha=0.5 epsilon=0.2 (Gridworld)')
fig.savefig('./SARSA_sweep/gridworld/alpha0.5_epsilon0.2.png')
plt.clf()

#alpha = 0.5, epsilon = 0.3
fig = plt.figure()
plt.plot(range(len(grid_SARSA_data_5_3[2])), grid_SARSA_data_5_3[2])
plt.xlabel('Iterations')
plt.ylabel('Total Discounted Rewards')
plt.title('SARSA Learning Curve: alpha=0.5 epsilon=0.3 (Gridworld)')
fig.savefig('./SARSA_sweep/gridworld/alpha0.3_epsilon0.3.png')
plt.clf()

#alpha = 0.6, epsilon = 0.1
fig = plt.figure()
plt.plot(range(len(grid_SARSA_data_6_1[2])), grid_SARSA_data_6_1[2])
plt.xlabel('Iterations')
plt.ylabel('Total Discounted Rewards')
plt.title('SARSA Learning Curve: alpha=0.6 epsilon=0.1 (Gridworld)')
fig.savefig('./SARSA_sweep/gridworld/alpha0.6_epsilon0.1.png')
plt.clf()


if not os.path.isdir('./SARSA_sweep/pendulum'):
    os.makedirs('./SARSA_sweep/pendulum')


#alpha = 0.4, epsilon = 0.1
fig = plt.figure()
plt.plot(range(len(pendulum_SARSA_data_4_1[2])), pendulum_SARSA_data_4_1[2])
plt.xlabel('Iterations')
plt.ylabel('Total Discounted Rewards')
plt.title('SARSA Learning Curve: alpha=0.4 epsilon=0.1 (pendulum)')
fig.savefig('./SARSA_sweep/pendulum/alpha0.4_epsilon0.1.png')
plt.clf()

#alpha = 0.5, epsilon = 0.1
fig = plt.figure()
plt.plot(range(len(pendulum_SARSA_data_5_1[2])), pendulum_SARSA_data_5_1[2])
plt.xlabel('Iterations')
plt.ylabel('Total Discounted Rewards')
plt.title('SARSA Learning Curve: alpha=0.5 epsilon=0.1 (pendulum)')
fig.savefig('./SARSA_sweep/pendulum/alpha0.5_epsilon0.1.png')
plt.clf()

#alpha = 0.5, epsilon = 0.2
fig = plt.figure()
plt.plot(range(len(pendulum_SARSA_data_5_2[2])), pendulum_SARSA_data_5_2[2])
plt.xlabel('Iterations')
plt.ylabel('Total Discounted Rewards')
plt.title('SARSA Learning Curve: alpha=0.5 epsilon=0.2 (pendulum)')
fig.savefig('./SARSA_sweep/pendulum/alpha0.5_epsilon0.2.png')
plt.clf()

#alpha = 0.5, epsilon = 0.3
fig = plt.figure()
plt.plot(range(len(pendulum_SARSA_data_5_3[2])), pendulum_SARSA_data_5_3[2])
plt.xlabel('Iterations')
plt.ylabel('Total Discounted Rewards')
plt.title('SARSA Learning Curve: alpha=0.5 epsilon=0.3 (pendulum)')
fig.savefig('./SARSA_sweep/pendulum/alpha0.3_epsilon0.3.png')
plt.clf()

#alpha = 0.6, epsilon = 0.1
fig = plt.figure()
plt.plot(range(len(pendulum_SARSA_data_6_1[2])), pendulum_SARSA_data_6_1[2])
plt.xlabel('Iterations')
plt.ylabel('Total Discounted Rewards')
plt.title('SARSA Learning Curve: alpha=0.6 epsilon=0.1 (pendulum)')
fig.savefig('./SARSA_sweep/pendulum/alpha0.6_epsilon0.1.png')
plt.clf()






#ALPHA-EPSILON SWEEP(Q-Learning)
if not os.path.isdir('./Q_Learning_sweep/gridworld/'):
    os.makedirs('./Q_Learning_sweep/gridworld/')

#alpha = 0.4, epsilon = 0.1
fig = plt.figure()
plt.plot(range(len(grid_Q_Learning_data_4_1[2])), grid_Q_Learning_data_4_1[2])
plt.xlabel('Iterations')
plt.ylabel('Total Discounted Rewards')
plt.title('Q-Learning Learning Curve: alpha=0.4 epsilon=0.1 (Gridworld)')
fig.savefig('./Q_Learning_sweep/gridworld/alpha0.4_epsilon0.1.png')
plt.clf()

#alpha = 0.5, epsilon = 0.1
fig = plt.figure()
plt.plot(range(len(grid_Q_Learning_data_5_1[2])), grid_Q_Learning_data_5_1[2])
plt.xlabel('Iterations')
plt.ylabel('Total Discounted Rewards')
plt.title('Q-Learning Learning Curve: alpha=0.5 epsilon=0.1  (Gridworld)')
fig.savefig('./Q_Learning_sweep/gridworld/alpha0.5_epsilon0.1.png')
plt.clf()

#alpha = 0.5, epsilon = 0.2
fig = plt.figure()
plt.plot(range(len(grid_Q_Learning_data_5_2[2])), grid_Q_Learning_data_5_2[2])
plt.xlabel('Iterations')
plt.ylabel('Total Discounted Rewards')
plt.title('Q-Learning Learning Curve: alpha=0.5 epsilon=0.2  (Gridworld)')
fig.savefig('./Q_Learning_sweep/gridworld/alpha0.5_epsilon0.2.png')
plt.clf()

#alpha = 0.5, epsilon = 0.3
fig = plt.figure()
plt.plot(range(len(grid_Q_Learning_data_5_3[2])), grid_Q_Learning_data_5_3[2])
plt.xlabel('Iterations')
plt.ylabel('Total Discounted Rewards')
plt.title('Q-Learning Learning Curve: alpha=0.5 epsilon=0.3  (Gridworld)')
fig.savefig('./Q_Learning_sweep/gridworld/alpha0.5_epsilon0.3.png')
plt.clf()

#alpha = 0.6, epsilon = 0.1
fig = plt.figure()
plt.plot(range(len(pendulum_Q_Learning_data_6_1[2])), pendulum_Q_Learning_data_6_1[2])
plt.xlabel('Iterations')
plt.ylabel('Total Discounted Rewards')
plt.title('Q-Learning Learning Curve: alpha=0.6 epsilon=0.1  (Gridworld)')
fig.savefig('./Q_Learning_sweep/gridworld/alpha0.6_epsilon0.1.png')
plt.clf()


if not os.path.isdir('./Q_Learning_sweep/pendulum/'):
    os.makedirs('./Q_Learning_sweep/pendulum/')

#alpha = 0.4, epsilon = 0.1
fig = plt.figure()
plt.plot(range(len(pendulum_Q_Learning_data_4_1[2])), pendulum_Q_Learning_data_4_1[2])
plt.xlabel('Iterations')
plt.ylabel('Total Discounted Rewards')
plt.title('Q-Learning Learning Curve: alpha=0.4 epsilon=0.1 (pendulum)')
fig.savefig('./Q_Learning_sweep/pendulum/alpha0.4_epsilon0.1.png')
plt.clf()

#alpha = 0.5, epsilon = 0.1
fig = plt.figure()
plt.plot(range(len(pendulum_Q_Learning_data_5_1[2])), pendulum_Q_Learning_data_5_1[2])
plt.xlabel('Iterations')
plt.ylabel('Total Discounted Rewards')
plt.title('Q-Learning Learning Curve: alpha=0.5 epsilon=0.1  (pendulum)')
fig.savefig('./Q_Learning_sweep/pendulum/alpha0.5_epsilon0.1.png')
plt.clf()

#alpha = 0.5, epsilon = 0.2
fig = plt.figure()
plt.plot(range(len(pendulum_Q_Learning_data_5_2[2])), pendulum_Q_Learning_data_5_2[2])
plt.xlabel('Iterations')
plt.ylabel('Total Discounted Rewards')
plt.title('Q-Learning Learning Curve: alpha=0.5 epsilon=0.2  (pendulum)')
fig.savefig('./Q_Learning_sweep/pendulum/alpha0.5_epsilon0.2.png')
plt.clf()

#alpha = 0.5, epsilon = 0.3
fig = plt.figure()
plt.plot(range(len(pendulum_Q_Learning_data_5_3[2])), pendulum_Q_Learning_data_5_3[2])
plt.xlabel('Iterations')
plt.ylabel('Total Discounted Rewards')
plt.title('Q-Learning Learning Curve: alpha=0.5 epsilon=0.3  (pendulum)')
fig.savefig('./Q_Learning_sweep/pendulum/alpha0.5_epsilon0.3.png')
plt.clf()

#alpha = 0.6, epsilon = 0.1
fig = plt.figure()
plt.plot(range(len(pendulum_Q_Learning_data_6_1[2])), pendulum_Q_Learning_data_6_1[2])
plt.xlabel('Iterations')
plt.ylabel('Total Discounted Rewards')
plt.title('Q-Learning Learning Curve: alpha=0.6 epsilon=0.1  (pendulum)')
fig.savefig('./Q_Learning_sweep/pendulum/alpha0.6_epsilon0.1.png')
plt.clf()





#Learning Curves
if not os.path.isdir('./learning_curves/'):
    os.makedirs('./learning_curves/')

#Value Iteration
fig = plt.figure()
plt.plot(value_iteration_data[3], value_iteration_data[2])
plt.xlabel('Iterations')
plt.ylabel('Mean Value Function')
plt.title('Value Iteration Learning Curve')
fig.savefig('./learning_curves/Value Iteration.png')
plt.clf()

# Policy Iteration
fig = plt.figure()
plt.plot(policy_iteration_data[3], policy_iteration_data[2])
plt.xlabel('Iterations')
plt.ylabel('Mean Value Function')
plt.title('Value Iteration Learning Curve')
fig.savefig('./learning_curves/Policy Iteration.png')
plt.clf()

#SARSA-Gridworld
fig = plt.figure()
plt.plot(range(len(grid_SARSA_data_5_1[2])), grid_SARSA_data_5_1[2])
plt.xlabel('Iterations')
plt.ylabel('Total Discounted Rewards')
plt.title('SARSA Learning Curve (Gridworld)')
fig.savefig('./learning_curves/SARSA(gridworld).png')
plt.clf()

#Q Learning-Gridworld
fig = plt.figure()
plt.plot(range(len(grid_Q_Learning_data_5_1[2])), grid_Q_Learning_data_5_1[2])
plt.xlabel('Iterations')
plt.ylabel('Total Discounted Rewards')
plt.title('Q-Learning Learning Curve (Gridworld)')
fig.savefig('./learning_curves/Q-Learning(gridworld).png')
plt.clf()

#SARSA-Pendulum
fig = plt.figure()
plt.plot(range(len(pendulum_SARSA_data_5_1[2])), pendulum_SARSA_data_5_1[2])
plt.xlabel('Iterations')
plt.ylabel('Total Discounted Rewards')
plt.title('SARSA Learning Curve (Pendulum)')
fig.savefig('./learning_curves/SARSA(pendulum).png')
plt.clf()

#Q Learning-Pendulum
fig = plt.figure()
plt.plot(range(len(pendulum_Q_Learning_data_5_1[2])), pendulum_Q_Learning_data_5_1[2])
plt.xlabel('Iterations')
plt.ylabel('Total Discounted Rewards')
plt.title('Q-Learning Learning Curve (Pendulum)')
fig.savefig('./learning_curves/Q-Learning(pendulum).png')
plt.clf()

               
pdb.set_trace()