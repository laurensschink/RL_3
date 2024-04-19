import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from itertools import count
import numpy as np


# Combined example from https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py
# With tips from: https://stats.stackexchange.com/questions/336531/how-to-initialize-policy-for-mountain-car-problem


class Policy(nn.Module):

    def __init__(self, n_observations, n_actions,layer1=16,layer2=16,activation=F.leaky_relu):
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(n_observations, layer1)
        self.layer2 = nn.Linear(layer1, layer2)
        self.layer3 = nn.Linear(layer2, n_actions)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return F.softmax(self.layer3(x),dim=0)

class Agent():

    def __init__(self,policy, gamma = .99, eta=10.0):
        self.policy = policy
        self.gamma = gamma
        self.eta = eta
        self.optimizer = optim.Adam(policy.parameters(), lr=1e-2)
        self.rewards = []
        self.saved_log_probs = []
#        self.eps = np.finfo(np.float32).eps.item()

    def calculate_entropy(self):
        H = 0
        for log_prob in self.saved_log_probs:
            H += log_prob.item()*np.exp(log_prob.item())
        return -1 * self.eta * H
        

    def select_action(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = self.policy(state)
        m = Categorical(action_probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def update_policy(self):
        R = 0
        policy_loss = []
        rewards = deque()
        for reward in self.rewards[::-1]:
            R = reward + self.gamma * R
            rewards.appendleft(R)
        rewards = torch.tensor(rewards)
#        returns = (rewards - rewards.mean()) / (rewards.std() + eps)
        entropy = self.calculate_entropy()
        for log_prob, R in zip(self.saved_log_probs, rewards):
            policy_loss.append(-log_prob * R + entropy)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        print(self.saved_log_probs)
        del self.rewards[:]
        del self.saved_log_probs[:]


def main():

    env = gym.make('MountainCar-v0')
    eval_env = gym.make('MountainCar-v0',render_mode='human')

    action_space = env.action_space.n
    observation_space = env.observation_space.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    policy_net = Policy(observation_space, action_space).to(device)
    agent = Agent(policy_net)
    running_reward = 10
    for i_episode in count(1):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = agent.select_action(state)
            state, reward, done, truncated, _ = env.step(action)
            reward = reward + np.abs(state[1]) # add speed to reward to encourage speeding up
            if state[0] > -.2:
                reward += 1 # add reward of 1 for reaching up high enough up the slope
            agent.rewards.append(reward)
            ep_reward += reward
            if truncated:
                #print('episode truncated')
                break
            if done: # don't reset trace after truncation, not sure if this is correct
                print('flag reached!')
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        agent.update_policy()
        if i_episode % 10 == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if (i_episode>5000):
            print("Max number of episodes reached! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break
        if done:
            print("Flag reached! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

if __name__ == '__main__':
    main()