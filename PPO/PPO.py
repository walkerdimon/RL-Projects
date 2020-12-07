import numpy as np 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from statistics import mean
import os
from os import listdir
from os.path import isfile, join
from networks import Actor, Critic
import pdb 


class PPO(object):
    def __init__(self, args, env):

        self.learning_rate = args.learning_rate
        self.gamma = args.gamma
        self.lamb = args.lamb
        self.batch_size = args.batch_size
        self.step = 0
        self.epochs = args.epochs
 
        self.actor = Actor()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        self.critic = Critic()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        self.env = env
        self.num_actions = env.num_actions
        self.num_states = env.num_states

        self.data = {'step' : [], 'reward' : [], 'losses' : []}



    def train(self):

        with torch.no_grad(): #no-grad makes computation faster
            batch = {'s' : [], 'a' : [], 'r' : [], 'w' : [], 'V_target' : [], 'pi' : []}

            for i in range(self.batch_size):
                traj = {'s' : [], 'a' : [], 'r' : [], 'V' : [], 'pi' : []}
                s = self.env.reset()
                done = False
                while done == False:
                    (mu, std) = self.actor(torch.from_numpy(s))
                    dist = torch.distributions.normal.Normal(mu, std)
                    a = dist.sample().numpy()
                    s1, r, done = self.env.step(a)
                    V = self.critic(torch.from_numpy(s)).item()
                    traj['s'].append(s)
                    traj['a'].append(a)
                    traj['r'].append(r)
                    traj['V'].append(V)
                    traj['pi'].append(dist.log_prob(torch.tensor(a)))
                    s = s1

                traj_len = len(traj['r'])
                r = np.append(traj['r'], 0.)
                V = np.append(traj['V'], 0.)
                delta = r[:-1] + (self.gamma * V[1:]) - V[:-1]
                A = delta.copy()

                for t in reversed(range(traj_len - 1)):
                    A[t] = A[t] + (self.gamma * self.lamb * A[t + 1])

                for t in reversed(range(traj_len)):
                    V[t] = r[t] + (self.gamma * V[t + 1])

                V = V[:-1]

                batch['s'].extend(traj['s'])
                batch['a'].extend(traj['a'])
                batch['r'].extend(traj['r'])
                batch['w'].extend(A)
                batch['V_target'].extend(V)
                batch['pi'].extend(traj['pi'])

            batch['num_steps'] = len(batch['r'])
            batch['s'] = torch.tensor(batch['s'], requires_grad=False, dtype=torch.double)
            batch['a'] = torch.tensor(batch['a'], requires_grad=False, dtype=torch.double)
            batch['r'] = torch.tensor(batch['r'], requires_grad=False, dtype=torch.double)
            batch['w'] = torch.tensor(batch['w'], requires_grad=False, dtype=torch.double)
            batch['V_target'] = torch.tensor(batch['V_target'], requires_grad=False, dtype=torch.double)
            batch['pi'] = torch.tensor(batch['pi'], requires_grad=False, dtype=torch.double)


        with torch.no_grad():
            N = batch['r'].shape[0] / self.batch_size

        #optimize Actor network
        for actor_epoch in range(10):
            self.actor_optimizer.zero_grad()
            (mu, std) = self.actor(batch['s'])
            dist = torch.distributions.normal.Normal(mu, std)
            pi = dist.log_prob(batch['a']).sum(axis=-1)
            ratio = torch.exp(pi - batch['pi'])
            surrogate = ratio * batch['w']
            clipped = torch.clamp(ratio, min= 1 - 0.2, max = 1 + 0.2) * batch['w']
            loss = - torch.mean((torch.min(surrogate, clipped)))
            loss.backward()
            self.actor_optimizer.step()

        #optimize Critic network
        for critic_epoch in range(10):
            self.critic_optimizer.zero_grad()
            V = self.critic(batch['s'])
            loss = nn.MSELoss()(V.squeeze(1), batch['V_target'])
            loss.backward()
            self.critic_optimizer.step()
        self.data['losses'].append(loss.item())


        #logging
        self.step += batch['r'].shape[0]
        self.data['step'].append(self.step)
        self.data['reward'].append(batch['r'].mean() * N)


    def save_model(self):

        if not os.path.exists('./model_save/'):
            os.makedirs('./model_save/')

        save_dir = './model_save/' + str(self.epochs) + '_epochs.pt'

        torch.save({'data': self.data,
                    'actor': self.actor,
                    'actor_optim': self.actor_optimizer,
                    'critic_optim': self.critic_optimizer,
                    'critic': self.critic}, save_dir)

    
    def generate_results(self):

        #LOAD MODEL        
        if not os.path.isdir('./results/'):
            os.makedirs('./results/')

        model_steps = np.array(self.data['step'])
        model_rewards = np.array(self.data['reward'])
        model_losses = np.array(self.data['losses'])

        #LEARNING CURVES
        print('Generating Learning Curves...')
        fig = plt.figure()
        plt.plot(model_steps, model_rewards)
        plt.xlabel('Simulation Steps')
        plt.ylabel('Total Reward')
        plt.title('Actor Learning Curve')
        plt.savefig('./results/Actor_Learning_Curve_' + str(self.epochs) + '_Epochs.png')

        fig = plt.figure()
        plt.plot(model_steps, model_losses)
        plt.xlabel('Simulation Steps')
        plt.ylabel('Critic Loss')
        plt.title('Critic Learning Curve')
        plt.savefig('./results/Critic_Learning_Curve_' + str(self.epochs) + '_Epochs.png')
  

        #EXAMPLE TRAJECTORY
        print('Generating Example Trajectory...')
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
            (mu, std) = self.actor(torch.from_numpy(s))
            dist = torch.distributions.normal.Normal(mu, std)
            a = dist.sample().numpy()
            s, r, done = self.env.step(a)
            data['t'].append(data['t'][-1] + 1)
            data['s'].append(s)
            data['a'].append(a)
            data['r'].append(r)
        # Parse data from simulation
        data['s'] = np.array(data['s'])
        theta = data['s'][:, 0]
        thetadot = data['s'][:, 1]
        # Plot data and save to png file
        fig = plt.figure()
        plt.plot(data['t'], theta, label='theta')
        plt.plot(data['t'], thetadot, label='thetadot')
        plt.legend()
        plt.savefig('./results/Example_Trajectory_' + str(self.epochs) + '_Epochs.png' )


        #ANIMATED TRAJECTORY
        print('Generating Animated Tragjectory...')
        filename='./results/Animated_Trajectory_' + str(self.epochs) + '_Epochs.gif'
        writer='imagemagick'
        s = self.env.reset()
        s_traj = [s]
        done = False
        while not done:
            (mu, std) = self.actor(torch.from_numpy(s))
            dist = torch.distributions.normal.Normal(mu, std)
            a = dist.sample().numpy()
            s, r, done = self.env.step(a)
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


        #POLICY VISUALIZATION
        print('Generating Policy Visualization...')
        theta_range = np.linspace(-np.pi, np.pi, 200)
        theta_dot_range = np.linspace(-self.env.max_thetadot_for_init, self.env.max_thetadot_for_init, 200)
        policy = np.zeros((len(theta_range), len(theta_dot_range)))
        for i in range(len(theta_range)):
            for j in range(len(theta_dot_range)):
                state = torch.tensor([theta_range[i], theta_dot_range[j]], dtype=torch.float64)
                (mu, std) = self.actor(state)
                dist = torch.distributions.normal.Normal(mu, std)
                a = dist.sample().numpy()
                policy[i][j] = a
        fig = plt.figure()
        plt.imshow(policy, cmap='coolwarm')
        plt.xlabel('theta dot')
        plt.ylabel('theta')
        plt.colorbar()
        plt.title('Policy Visualization')
        fig.savefig('./results/Policy_Visualization_' + str(self.epochs) + '_Epochs.png')
        fig.clf()


        #VALUE FUNCTION VISUALIZATION
        print('Generating Value Function Visualization...')
        theta_range = np.linspace(-np.pi, np.pi, 200)
        theta_dot_range = np.linspace(-self.env.max_thetadot_for_init, self.env.max_thetadot_for_init, 200)
        value = np.zeros((len(theta_range), len(theta_dot_range)))
        for i in range(len(theta_range)):
            for j in range(len(theta_dot_range)):
                state = torch.tensor([theta_range[i], theta_dot_range[j]], dtype=torch.float64)
                V = self.critic(state).item()
                value[len(theta_range)-i][j] = V
        fig = plt.figure()
        plt.imshow(value, cmap='coolwarm')
        plt.xlabel('theta dot')
        plt.ylabel('theta')
        plt.colorbar()
        plt.title('Value Function Visualization')
        fig.savefig('./results/Value_Visualization_' + str(self.epochs) + '_Epochs.png')
        fig.clf()


        print('done')