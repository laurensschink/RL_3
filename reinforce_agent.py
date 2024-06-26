
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
    def __init__(self,action_space,observation_space):
        super(Policy, self).__init__()
        self.lin1= nn.Linear(observation_space, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.lin2 = nn.Linear(128, action_space)

    def forward(self, x):
        x = self.lin1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.lin2(x)
        return F.softmax(action_scores, dim=1)

class Reinforce():
    def __init__(self,n_actions,n_observations,gamma,lr=1e-3,eta=1e-3):
        self.eta = eta
        self.gamma = gamma
        self.policy = Policy(n_actions,n_observations)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.eps = np.finfo(np.float32).eps.item()
        self.saved_log_probs = []
        self.entropies = []
        self.rewards = []
        self.grad_log = []

    def select_action(self,state):
#        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.entropies.append(m.entropy())
        self.saved_log_probs.append(m.log_prob(action))
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
        
        # store gradients
        grads = []
        for param in self.policy.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1).detach().cpu().numpy()**2) # get squared grads https://discuss.pytorch.org/t/should-it-really-be-necessary-to-do-var-detach-cpu-numpy/35489
        self.grad_log.append(np.concatenate(grads))

        del self.entropies[:]
        del self.saved_log_probs[:]
        del self.rewards[:]
        
        # extra functions for the gradient plot
    def get_gradients(self):
        return self.grad_log

    def reset_gradients(self):
        self.grad_log = []

