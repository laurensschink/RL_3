
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import numpy as np

import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt


class Policy(nn.Module):

    def __init__(self, n_observations, n_actions,layer1=16,layer2=16,activation=F.leaky_relu, output_activation=torch.nn.Softmax(dim=0)):
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(n_observations, layer1)
        self.layer2 = nn.Linear(layer1, layer2)
        self.layer3 = nn.Linear(layer2, n_actions)
        self.activation = activation
        self.output_activation = output_activation

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.output_activation(self.layer3(x))
        return x

class Ac3():

    def __init__(self,n_actions,n_observations,gamma = .99, lr=1e-3, eta=.1):
        self.policy = Policy(n_observations, n_actions)
        self.gamma = gamma
        self.eta = eta
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.rewards = []
        self.saved_log_probs = []

    def calculate_entropy(self):
        #Entropy defined as sum(p(s)log(p(s))), from the lecture slides
        #Used for exploration
        H = 0
        for log_prob in self.saved_log_probs:   # -> i would guess we can avoid this loop
            H += log_prob.item()*np.exp(log_prob.item())  # by calculating with tensors
        return -1 * self.eta * H
        

    def select_action(self,state):
        action_probs = self.policy(state)   
        m = Categorical(action_probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def update_policy(self, value_func=None):
        R = 0
        policy_loss = []
        rewards = deque()
        for reward in self.rewards[::-1]:
            #Update rewards
            R = reward + self.gamma * R
            rewards.appendleft(R)
        rewards = torch.tensor(rewards)
        entropy = self.calculate_entropy()
        for log_prob, R in zip(self.saved_log_probs, rewards):
            if value_func==None:
                #Pure REINFORCE policy update
                policy_loss.append(-log_prob * R + entropy)
            else:
                #Update policy with value function
                policy_loss.append(-log_prob*value_func + entropy)
        
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        del self.rewards[:]
        del self.saved_log_probs[:]

    def update_value_function(self, state_vals, Q_vals):
        #Updates value function using mean square error between state and Q-values
        loss_function = nn.MSELoss()
        loss = loss_function(state_vals, Q_vals)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()