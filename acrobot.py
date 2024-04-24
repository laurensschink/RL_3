import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from itertools import count
import numpy as np
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

# Combined example from https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py
# With tips from: https://stats.stackexchange.com/questions/336531/how-to-initialize-policy-for-mountain-car-problem
# Actor critic algorithms implementation inspired by https://medium.com/@ym1942/policy-gradient-methods-from-reinforce-to-actor-critic-d56ff0f0af0a

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

class Agent():

    def __init__(self,policy, gamma = .99, eta=1.0, lr=1e-2):
        self.policy = policy
        self.gamma = gamma
        self.eta = eta
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.rewards = []
        self.saved_log_probs = []
#        self.eps = np.finfo(np.float32).eps.item()

    def calculate_entropy(self):
        H = 0
        for log_prob in self.saved_log_probs:
            H += log_prob.item()*np.exp(log_prob.item())
        return -1 * self.eta * H
        

    def select_action(self,state):
        action_probs = self.policy(state)
        m = Categorical(action_probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def update_policy(self, advantage=None):
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
            if advantage==None:
                policy_loss.append(-log_prob * R + entropy)
            else:
                policy_loss.append(-log_prob*advantage + entropy)
        
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        del self.rewards[:]
        del self.saved_log_probs[:]

    def update_value_function(self, state_vals, Q_vals):
        loss_function = nn.MSELoss()
        loss = loss_function(state_vals, Q_vals)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def smooth(y, window, poly=2):
    return savgol_filter(y, window, poly)

def main(bootstrap=True, n_step=5, gamma=.99, lr=1e-2, eta=10.0):
    env = gym.make('Acrobot-v1')
    eval_env = gym.make('Acrobot-v1',render_mode='human')

    action_space = env.action_space.n
    observation_space = env.observation_space.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    policy_net = Policy(observation_space, action_space).to(device)
    agent = Agent(policy_net, gamma=gamma, lr=lr, eta=eta)
    running_reward = -100
    if bootstrap:
        value_function = Policy(observation_space, 1, output_activation=torch.nn.ReLU()).to(device)
        value_agent = Agent(value_function, gamma=gamma, lr=lr, eta=eta)

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
            #reward = reward + np.abs(state[1]) # add speed to reward to encourage speeding up
            #if state[0] > -.2:
            #    reward += 1 # add reward of 1 for reaching up high enough up the slope
            agent.rewards.append(reward)
            #state = torch.from_numpy(state).float().unsqueeze(0)
            #trace_states.append(state)
            ep_reward += reward
            if truncated:
                #print('episode truncated')
                T = t
                break
            if done: # don't reset trace after truncation, not sure if this is correct
                print('Goal reached!')
                T = t
                break

        if bootstrap:    
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
            agent.update_policy(advantages)
        else:
            agent.update_policy()

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        episode_rewards.append(ep_reward)
        if i_episode % 10 == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if (i_episode>500):
            print("Max number of episodes reached! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break
        #if done:
        #    print("Goal reached! Running reward is now {} and "
        #          "the last episode runs to {} time steps!".format(running_reward, t))
        #    break
    
    #Plot results
    episode_rewards = smooth(episode_rewards, 3)
    eps = range(i_episode)
    plt.plot(eps, episode_rewards)
    plt.title('Reward Function')
    plt.xlabel('Episode')
    plt.ylabel('Rolling Average of Reward')
    plt.show()

    env.close()
if __name__ == '__main__':
    main()