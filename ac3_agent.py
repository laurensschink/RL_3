import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import numpy as np

torch.autograd.set_detect_anomaly(True) 
class Policy(nn.Module):

    def __init__(self, n_observations, n_actions,layer1=32,layer2=16,activation=F.leaky_relu):
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
        return probs

class Q_value(nn.Module):

    def __init__(self, n_observations, layer1=16,layer2=16,activation=nn.Tanh()):
        super(Q_value, self).__init__()

        # action selection network layers
        self.layer1 = nn.Linear(n_observations, layer1)
        self.layer2 = nn.Linear(layer1, layer2)
        self.layer3 = nn.Linear(layer2, 1)
        self.activation = activation

    def forward(self,x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        value = self.layer3(x)
        return value

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, n_observations, n_actions,layer1=32,layer2=16,activation=F.leaky_relu):
        super(Policy, self).__init__()
        self.lin1 = nn.Linear(n_observations, layer1)
        self.lin2 = nn.Linear(layer1, layer2)
        self.activation = activation
        # actor's layer
        self.action_head = nn.Linear(layer2, 2)
        # critic's layer
        self.value_head = nn.Linear(layer2, 1)

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = self.activation(self.lin1(x))
        x = self.activation(self.lin2(x))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_prob, state_values


class Ac3():

    def __init__(self,n_actions,n_observations,
                    bootstrap=True, baseline=True, 
                    n_step=5,gamma = .99, lr=1e-3, eta=.1):
        self.policy = Policy(n_observations, n_actions)
#        self.q_value = Q_value(n_observations)
        self.bootstrap = bootstrap
        self.baseline = baseline
        self.n_step = n_step
        self.gamma = gamma
        self.eta = eta
        self.pol_optim = optim.Adam(self.policy.parameters(), lr=lr)
 #       self.q_optim = optim.Adam(self.q_value.parameters(), lr=lr)
        self.states = []
        self.rewards = []
        self.actions = []
        self.saved_log_probs = []
        self.values = []

    def select_action(self,state):
        self.states.append(state)
        probs, value = self.policy(state)    # action 
        distr  = Categorical(probs)   # action 
        action = distr.sample()       # action
#        value = self.q_value(state)   # value 
        self.saved_log_probs.append(distr.log_prob(action)) 
        self.values.append(value)                           
        return action.item()          # action

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
#                    value = self.q_value(self.states[i+n_step]).detach()
                    _,value = self.policy(self.states[i+n_step])
                n_state_value = (self.gamma**n_step) * value    # -> Here i should decouple the number from the tensor incl gradien
                R = reward + torch.stack([(self.gamma**n) * torch.tensor(self.rewards[i+n]) for n in range(n_step)]).sum() + n_state_value
            else:
                R = reward + self.gamma * R
            rewards.appendleft(R)

        rewards = torch.tensor(rewards)         # is this the best way to convert a deque into a tensor?
 
#        entropy = self.calculate_entropy()
        for log_prob,value, R in zip(self.saved_log_probs,self.values, rewards):
            entropy = log_prob * torch.exp(log_prob)
            if self.baseline:
                advantage = R - value
                policy_losses.append(-log_prob * advantage + self.eta * entropy) # is this the correct loss?!
            else:
                policy_losses.append(-log_prob * value + self.eta * entropy) # is this the correct loss?!

            value_losses.append(F.smooth_l1_loss(value.squeeze(), R)) # should we add entropy here too?

        self.pol_optim.zero_grad()      
#        self.q_optim.zero_grad()  
        policy_loss = torch.stack(policy_losses).sum() 
        value_loss = torch.stack(value_losses).sum()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()    
        loss.backward()
#        value_loss.backward()
        self.pol_optim.step()
#        self.q_optim.step()

        del self.rewards[:]
        del self.saved_log_probs[:]