# What is Included
Found here is a repository of the REINFORCE algorithm reveloped and tested in a discrete action gridworld environment. The goal of this project was to implement the REINFORCE algorithm, as well as several extensions (causality, baseline shift, and importance sampling) to derive the optimal policy for the environment. The derived optimal policy, as well as learning curves of the extensions are below.

From Reinforcement Learning Course taught at The University of Illinois, Fall 2020.

## TO RUN CODE:
The main run file is found in "run.py". The Reinforce algorithm as well as all extensions is found in "REINFORCE.py" To run the code, execute 'run.py --*EXTENSION TYPE*)' in the terminal, where *EXTENSION TYPE* is either 'base_model' (no extension), 'baseline_causality' (baseline shift + causality), 'baseline_importance' (baseline shift + importance sampling), 'causality_importance' (causality + importance sampling), or 'baseline_causality_importance' (baseline shift + causality + importance sampling). The default number of episodes is set to 10,000 but can be changed with the '-- episodes' argument.

### Base Model Learning Curve: Model trained for 10,000 Episodes
![alt text](https://github.com/compdyn/598rl-fa20/blob/hw3_wdimon2/hw3/hw3_wdimon2/generated_results/base_learning_curve_10000_episodes.png)

### Base Model Optimal Policy: Model trained for 10,000 Episodes
![alt text](https://github.com/compdyn/598rl-fa20/blob/hw3_wdimon2/hw3/hw3_wdimon2/generated_results/SGD_Base_Policy_10000_Episodes.png)

### Learning Curve Comparison of All Extensions: Model trained for 10,000 Episodes
![alt text](https://github.com/compdyn/598rl-fa20/blob/hw3_wdimon2/hw3/hw3_wdimon2/generated_results/comparison_learning_curve_10000_episodes.png)

