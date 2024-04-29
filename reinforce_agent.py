import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
# Test for bugs (will be plenty of errors to iron out)
# add entropy regularization
# make hyper parameters configurable

class Policy(nn.Module):
    def __init__(self,action_space,observation_space, hidden_nodes, dropout):
        super(Policy, self).__init__()
        self.lin1= nn.Linear(observation_space, hidden_nodes)
        self.dropout = nn.Dropout(p=dropout)
        self.lin2 = nn.Linear(hidden_nodes, action_space)

    def forward(self, x):
        x = self.lin1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.lin2(x)
        probs = F.softmax(action_scores, dim=1)
        distr  = Categorical(probs)       
        return distr

class Reinforce():
    def __init__(self,n_actions,n_observations,gamma, lr=1e-3,
                eta=1e-3, hidden_nodes = 32, dropout = 0.3):
        self.eta = eta
        self.gamma = gamma
        self.policy = Policy(n_actions,n_observations, hidden_nodes,dropout)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.eps = np.finfo(np.float32).eps.item()
        self.states = []
        self.saved_log_probs = []
        self.rewards = []
        self.entropies = []

    def select_action(self,state):
#        state = torch.from_numpy(state).float().unsqueeze(0)
        distr = self.policy(state)
        action = distr.sample()
        self.entropies.append(distr.entropy().mean())
        self.saved_log_probs.append(distr.log_prob(action))
        return action.item()

    def update_policy(self):
        R = 0
        policy_loss = []
        returns = deque()
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.appendleft(R)
        returns = torch.tensor(returns)
        entropy = torch.tensor(self.entropies)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R + self.eta * log_prob * torch.exp(log_prob))
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum() 
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]
    
    def exploit(self, state):
        with torch.no_grad():
            distr = self.policy(state)
            return torch.argmax(distr).item()
        

