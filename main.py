import gymnasium as gym
import torch
import torch.nn as nn
from itertools import count

import numpy as np # -> Todo: switch to torch

import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt

# import agents
from reinforce_agent import Reinforce
from ac3_agent import Ac3, Policy



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
    agent = Ac3(n_actions=action_space,n_observations=observation_space,gamma=gamma, lr=lr, eta=eta)
    running_reward = -100
    if bootstrap or baseline:
        value_function = Policy(observation_space, 1, output_activation=nn.ReLU()).to(device)
        value_agent = Ac3( n_actions=action_space,n_observations=observation_space, gamma=gamma, lr=lr, eta=eta)
        value_agent.policy = value_function


    episode_rewards = []
    for i_episode in count(1):
        state, _ = env.reset()
        ep_reward = 0
        trace_states = []
        Q_trace = []
        for t in range(1, 10000):  # Don't infinite loop while learning
            state = torch.from_numpy(state).float().unsqueeze(0)
            trace_states.append(state)
            action = agent.select_action(state)
            state, reward, done, truncated, _ = env.step(action)
            agent.rewards.append(reward)
            
            ep_reward += reward
            if truncated:
                T = t
                break
            if done:
#                print('Goal reached!')
                T = t
                break

        if bootstrap or baseline:
            if not bootstrap:
                n_step = 0
            for t in range(T):
                Q = 0
                if t+n_step > T:
                    n_step_net = T-t
                else:
                    n_step_net = n_step
                    V_end = value_function.forward(trace_states[t]).item()
                    Q += gamma**n_step_net * V_end

                for k in range(n_step_net):
                    Q += gamma**k * agent.rewards[t+k]
                Q_trace.append(Q)
            
            trace_states = np.vstack(trace_states).astype(float)
            trace_states = torch.FloatTensor(trace_states).to(device)
            state_values = value_function(trace_states).to(device)
            Q_trace = torch.tensor(Q_trace).view(-1,1)
            with torch.no_grad():
                advantages = Q_trace - state_values
            value_agent.update_value_function(state_values, Q_trace)
            if baseline:
                agent.update_policy(advantages)
            else:
                agent.update_policy(Q_trace)
        else:
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
    experiment(bootstrap=False, baseline=False, n_step=5, gamma=.99, lr=1e-3, eta=.1)

if __name__ == '__main__':
    main()