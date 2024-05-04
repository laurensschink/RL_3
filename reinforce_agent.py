import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque


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
          
        return probs

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
        self.grad_log = []

    def select_action(self,state):
#        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        distr  = Categorical(probs)     
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
        
        # store gradients
        grads = []
        for param in self.policy.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1).detach().cpu().numpy()**2) # get squared grads https://discuss.pytorch.org/t/should-it-really-be-necessary-to-do-var-detach-cpu-numpy/35489
        self.grad_log.append(np.concatenate(grads))

        del self.entropies[:]
        del self.saved_log_probs[:]
        del self.rewards[:]
    
    def exploit(self, state):
        with torch.no_grad():
            probs = self.policy(state)
            return torch.argmax(probs).item()
    
    
    # extra functions for the gradient plot
    def get_gradients(self):
        return self.grad_log

    def reset_gradients(self):
        self.grad_log = []
        

def experiment(gamma=.99, lr=1e-3, eta=.1, hidden_nodes=32, dropout=0.3,max_eps=801):
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
    eval_env = gym.make('Acrobot-v1')
    eval_rate = 25
    action_space = env.action_space.n
    observation_space = env.observation_space.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    agent = Reinforce(action_space,observation_space,gamma, lr=lr,
                        eta=eta, hidden_nodes = hidden_nodes, dropout = dropout)

    train_rewards, train_losses, eval_rewards = list(),list(),list()
    for _ in range(10):
        run_rewards,run_losses, = list(),list()
        run_eval_rewards = {}
        for i_episode in count(1):
            state, _ = env.reset()
            ep_reward = 0
            for t in range(1, 10000):  
                state = torch.from_numpy(state).float().unsqueeze(0)
                action = agent.select_action(state)
                state, reward, done, truncated, _ = env.step(action)
                agent.rewards.append(reward)
                ep_reward += reward
                if done or truncated:
                    break
            agent.update_policy()
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
                run_eval_rewards[i_episode] = np.mean(eval_list)
                agent.policy.train()     
            run_rewards.append(ep_reward)
            if (i_episode>max_eps):
                break