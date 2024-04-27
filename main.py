import gymnasium as gym
import torch
import torch.nn as nn
from itertools import count

import numpy as np # -> Todo: switch to torch

from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt

# import agents
from reinforce_agent import Reinforce
from ac3_agent import Ac3, Policy

def smooth(y, window, poly=2):
    #Helper function to smooth loss functions
    return savgol_filter(y, window, poly)


def experiment(bootstrap=False, baseline=False, n_step=5, gamma=.99, lr=1e-3, eta=.1):
    """
    Main function to run a policy-based deep reinforcement learning run on the acrobot environment.
    Acrobot is relatively reward-scarce, only gains any reward when reaching goal.
    It also terminates as soon as the goal is reached.
    Aim is therefore to reach the goal as quickly and consitently as possible.

    parameters:
    bootstrap (bool): Turns actor-critic n-step bootstrapping on
    baseline (bool): Turns actor critic baseline substraction on
    n_step (int): number of steps for n-step bootstrapping
    gamma (float): discount parameter
    lr (float): learning rate
    eta (float): entropy factor
    """
    env = gym.make('Acrobot-v1')
    eval_env = gym.make('Acrobot-v1',render_mode='human')

    max_eps = 2000
    action_space = env.action_space.n
    observation_space = env.observation_space.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    agent = Reinforce(n_actions=action_space,n_observations=observation_space,gamma=gamma, lr=lr, eta=eta)
#    agent = Ac3(n_actions=action_space,n_observations=observation_space,bootstrap=True, baseline=True, gamma=gamma, lr=lr, eta=eta)
    running_reward = -500

    episode_rewards = []
    for i_episode in count(1):
        state, _ = env.reset()
        ep_reward = 0
#        trace_states = []
#        Q_trace = []
        for t in range(1, 10000):  # Don't infinite loop while learning
            state = torch.from_numpy(state).float().unsqueeze(0)
#            trace_states.append(state)
            action = agent.select_action(state)
            state, reward, done, truncated, _ = env.step(action)
            agent.rewards.append(reward)
            ep_reward += reward
            if done or truncated:
                break

        agent.update_policy()

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        episode_rewards.append(ep_reward)
        if i_episode % 10 == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if (i_episode>max_eps):
            print("Max number of episodes reached! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


    #Plot results
    episode_rewards = smooth(episode_rewards, 3)
    eps = range(i_episode)
    plt.plot(eps, episode_rewards)
    plt.title('Reward Function')
    plt.xlabel('Episode')
    plt.ylabel('Rolling Average of Reward')
    plt.show()

    env.close()


def main():
    experiment(bootstrap=True, baseline=True, n_step=5, gamma=.99, lr=1e-3, eta=.1)

if __name__ == '__main__':
    main()