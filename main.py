import gymnasium as gym
import torch, pickle, os
import torch.nn as nn
from itertools import count 
from tqdm import tqdm

import numpy as np 

from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt

# import agents
from reinforce_agent import Reinforce


def smooth(y, window, poly=2):
    #Helper function to smooth loss functions
    return savgol_filter(y, window, poly)

torch.autograd.set_detect_anomaly(True)

def experiment(gamma=.99, lr=1e-3, eta=.1, dropout=0.3):
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
    #hyper_search_reinforce(50)
    experiment()

def hyper_search_reinforce(nr_searches):
    """hyperparameter search for the reinforce agent to find optimal the configuration.
    The hyperparameter space is defined in the dictionary search_space and with each run 
    a random combination of parameters is selected.

    Args:
        nr_searches (int): amount of searches to perform over the search space.
    """    
    env = gym.make('Acrobot-v1')
    eval_env = gym.make('Acrobot-v1')

    max_eps = 1501
    eval_rate = 100
    action_space = env.action_space.n
    observation_space = env.observation_space.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    search_space = {
    'hidden_nodes': [16, 32, 64, 128],
    'eta': [1, 0.1, 0.05 ,0.01, 0.005],
    'lr': [0.0001, 0.001, 0.01],
    'gamma': [0.9, 0.99, 0.999],
    'dropout': [0, .2, .4, .6]
    }    
    
    summaries = []
    
    for c,search in enumerate(range(nr_searches)):
        eval_rewards = {}
        episode_rewards = []
        param_dict = {key:np.random.choice(value) for key,value in search_space.items()}

        hidden_nodes = int(param_dict['hidden_nodes'])
        eta = param_dict['eta']
        lr = param_dict['lr']
        gamma = param_dict['gamma']
        dropout = param_dict['dropout']
        print(param_dict)

        agent = Reinforce(n_actions= action_space,n_observations= observation_space,
                        gamma= gamma, lr= lr, eta= eta, hidden_nodes= hidden_nodes, 
                        dropout= dropout)
#        for i_episode in count(1):
        for i_episode in tqdm(range(1,max_eps)):
            print(i_episode)
            state, _ = env.reset()
            ep_reward = 0 
            for t in range(1, 10000):  # learning loop
                state = torch.from_numpy(state).float().unsqueeze(0)
                action = agent.select_action(state)
                state, reward, done, truncated, _ = env.step(action)
                agent.rewards.append(reward)
                ep_reward += reward
                if done or truncated:
                    break
            agent.update_policy()
            episode_rewards.append(ep_reward)
            if i_episode % eval_rate == 0:    # evaluate learning progress
                eval_list = []
                agent.policy.eval()
                for i in range(10):
                    eval_reward = 0
                    state, _ = eval_env.reset()
                    while True:
                        state = torch.from_numpy(state).float().unsqueeze(0)
                        action = agent.exploit(state)
                        state, reward, done, truncated, _ = eval_env.step(action)
                        eval_reward+= reward
                        if done or truncated:
                            eval_list.append(eval_reward)
                            break
                eval_rewards[i_episode] = np.mean(eval_list)
                agent.policy.train()             
#            if (i_episode>max_eps):
#                break
        param_dict['avg_reward'] = np.mean(episode_rewards)
        param_dict['avg_eval_reward'] = np.mean(list(eval_rewards.values()))
        param_dict['learning_curve'] = episode_rewards
        param_dict['eval_curve'] = eval_rewards
        summaries.append(param_dict)
        print(f'run {c} - {nr_searches} completed.')
        print(f'mean reward: {param_dict["avg_reward"]:.2f}, mean eval: {param_dict["avg_eval_reward"]:.2f}')
        with open(os.path.join('results','grid_search_reinforce.pickle'), 'wb') as f:
            pickle.dump(summaries, f)

if __name__ == '__main__':
    main()