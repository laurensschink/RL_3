 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import numpy as np


class Policy(nn.Module):

    def __init__(self, n_observations, n_actions,layer1=16,layer2=16,activation=F.leaky_relu):
        super(Policy, self).__init__()

        # action selection network layers
        self.layer1 = nn.Linear(n_observations, layer1)
        self.layer2 = nn.Linear(layer1, layer2)
        self.layer3 = nn.Linear(layer2, n_actions)
        self.activation = activation
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self,x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        probs = self.softmax(self.layer3(x))
        distr  = Categorical(probs)        
        return distr

class Q_value(nn.Module):

    def __init__(self, n_observations, layer1=16,layer2=16,activation=nn.Tanh()):
        super(Q_value, self).__init__()

        # action selection network layers
        self.layer1 = nn.Linear(n_observations, layer1)
        self.layer2 = nn.Linear(layer1, layer2)
        self.layer3 = nn.Linear(layer2, 1)
        self.activation = activation

    def forward(self,x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        value = self.layer3(x)
        return value



class Ac3():

    def __init__(self,n_actions,n_observations,
                    bootstrap=True, baseline=True, 
                    n_step=5,gamma = .99, lr=1e-3, eta=.1):
        self.policy = Policy(n_observations, n_actions)
        self.q_value = Q_value(n_observations)
        self.bootstrap = bootstrap
        self.baseline = baseline
        self.n_step = n_step
        self.gamma = gamma
        self.eta = eta
        self.pol_optim = optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optim = optim.Adam(self.q_value.parameters(), lr=lr)
        self.states = []
        self.rewards = []
        self.actions = []
        self.saved_log_probs = []
        self.values = []

    def select_action(self,state):
        self.states.append(state)
        distr = self.policy(state)   
        value = self.q_value(state)
        action = distr.sample()
        self.saved_log_probs.append(distr.log_prob(action))
        self.values.append(value)
        return action.item()

    def update_policy(self, value_func=None):
        R = 0
        policy_losses = []
        value_losses = []
        rewards = deque()
        for i,reward in enumerate(self.rewards[::-1]):
            #Update rewards
            if self.bootstrap:
                n_step = min(self.n_step,len(self.rewards[::-1])-i)
                if i+n_step == len(self.rewards[::-1]):
                    value = 0
                else:
                    value = self.q_value(self.states[i+n_step])
                n_state_value = (self.gamma**n_step) * value
                R = reward + torch.stack([(self.gamma**n) * torch.tensor(self.rewards[i+n]) for n in range(n_step)]).sum() + n_state_value
            else:
                R = reward + self.gamma * R
            rewards.appendleft(R)

        rewards = torch.tensor(rewards)
 
#        entropy = self.calculate_entropy()
        for log_prob,value, R in zip(self.saved_log_probs,self.values, rewards):
            entropy = log_prob * torch.exp(log_prob)
            if self.baseline:
                advantage = R - value.item()
                policy_losses.append(-log_prob * advantage + self.eta * entropy)
            else:
                policy_losses.append(-log_prob * value.item() + self.eta * entropy)

            value_losses.append(F.smooth_l1_loss(value.squeeze(), R))

        self.pol_optim.zero_grad()      
        self.q_optim.zero_grad()  
        policy_loss = torch.stack(policy_losses).sum() 
        value_loss = torch.stack(value_losses).sum()
    
        policy_loss.backward(retain_graph=True)
        value_loss.backward(retain_graph=True)
        self.pol_optim.step()
        self.q_optim.step()

        del self.rewards[:]
        del self.saved_log_probs[:]

