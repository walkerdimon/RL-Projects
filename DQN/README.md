# What is Included
Found here is a repository of the DQN (Deep Q-Learning Network) algorithm developed and tested in a discrete action pendulum environment. The goal of this project was to train a neural network to balance an inverted pendulum. An animated GIF of the trained network is found below. From Reinforcement Learning Course taught at The University of Illinois, Fall 2020.

## TO RUN CODE:
main run file is found in "run.py". To train the network, execute "run.py --train" on the command line. This will initiate running the base model, with a memory size of 1000 experiences, and the target network updating every five steps. The memory size can be changed by adjusting the --memory_cap and --C_update arguments, respectively. After training is complete, the network model and data log will be saved to the "model_save" file. To generate results, execute "run.py --test". This will generate learning curves, example trajectories, value function visualization, and policy visualization. The ablation study was generated through executing "run.py --ablation" after running 3 training sessions of the base, no memory, no target, and no memory-no target models.

### Example Trajectory Gif: Model trained for 10,000 Epochs
![alt text](https://github.com/compdyn/598rl-fa20/blob/hw2_wdimon2/hw2/hw2_wdimon2/generated_results/Animated_Trajectory_10000_Episodes.gif)

