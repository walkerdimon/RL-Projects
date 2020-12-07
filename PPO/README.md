# What is Included
Found here is a repository of the PPO algorithm reveloped and tested in a continuous inverted pendulum environment. The goal of this project was to implement the PPO algorithm to train an agent to balance the pendulum upright. An example trajectory of the trained agent, as well as learning curves of the Actor and Critic networks are below.

From Reinforcement Learning Course taught at The University of Illinois, Fall 2020.

# HW4 for Walker Dimon (wdimon2)

## TO RUN CODE:
The main run file is found in "run.py". The PPO algorithm  is found in "PPO.py" To run the code, execute 'run.py' in the terminal. The default number of training epochs is set to 1,000 but can be changed with the '-- epochs' argument. Executing the 'run.py' command will run the PPO algorithm and train the actor and critic networks, save these networks along with a data log of critic loss, actor reward, and training steps (saved to the 'model_save' folder), and generate the requested results. These results are saved to the 'results' folder.

## Model Results
Training was conducted with using a batch-size of 50, a learning rate of 0.001, gamma of 0.99, and lambda of 0.95. During optimization of the actor and critic networks, importance sampling was performed for 10 loops.

### Actor Learning Curve: Model trained for 1,000 Epochs
![alt text](https://github.com/compdyn/598rl-fa20/blob/hw4_wdimon2/hw4/hw4_wdimon2/results/Actor_Learning_Curve_1000_Epochs.png)

### Critic Learning Curve: Model trained for 1,000 Epochs
![alt text](https://github.com/compdyn/598rl-fa20/blob/hw4_wdimon2/hw4/hw4_wdimon2/results/Critic_Learning_Curve_1000_Epochs.png)

### Example Trajectory: Model trained for 1,000 Epochs
![alt text](https://github.com/compdyn/598rl-fa20/blob/hw4_wdimon2/hw4/hw4_wdimon2/results/Example_Trajectory_1000_Epochs.png)

### Animated Trajectory: Model trained for 1,000 Epochs
![alt text](https://github.com/compdyn/598rl-fa20/blob/hw4_wdimon2/hw4/hw4_wdimon2/results/Animated_Trajectory_1000_Epochs.gif)



