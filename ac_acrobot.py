import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib
import matplotlib.pyplot as plt


# simple, seperate actor-critic nns
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state):
        action_probs = self.network(state)
        return action_probs

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, state):
        value = self.network(state)
        return value

# new entropy calculation 
def compute_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-6), dim=1)

# also seperate optimizers for both nns
def train(env, actor, critic, actor_optimizer, critic_optimizer, num_episodes, gamma=0.99, initial_entropy_weight=0.1, final_entropy_weight=0.001):
    loss_over_episodes = []
    rewards_per_episode = []
    for episode in range(num_episodes):
        
        # should find out if this is enough learning episodes, and if 0.1 is high enough
        entropy_weight = max(initial_entropy_weight - (initial_entropy_weight - final_entropy_weight) * (episode / 100), final_entropy_weight) # decrease to 0.001 in the first 100 epsiodes

        # init the environment, init lists
        state,_ = env.reset()
        state = torch.from_numpy(state).float().unsqueeze(0)
        log_probs = []
        values = []
        rewards = []
        entropies = []
        
        # done = False
        for t in range(1,10000):
            # sample an action
            probs = actor(state)
            dist = Categorical(probs)
            action = dist.sample()
            
            #get logbprob & entropy
            log_prob = dist.log_prob(action)
            entropy = compute_entropy(probs)
            
            # execute action
            next_state, reward, done, _, _ = env.step(action.item())
            next_state = torch.from_numpy(next_state).float().unsqueeze(0)
            
            # get critic estimate for reward for current state
            value = critic(state)
            
            #store data
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)
            
            state = next_state
            
        #returns and advantages
        returns = []
        advantages = []
        G = 0

        # reversely calculate returns and advantages from the episode
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            returns.append(G)
            advantages.append(G - values[t].item())
        
        returns = torch.tensor(returns, dtype=torch.float32).flip(dims=(0,))
        advantages = torch.tensor(advantages, dtype=torch.float32).flip(dims=(0,))
        
        # updating netwiorks
        # set grads to zero
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        
        # policy_loss.append(-log_prob * advantage + entropy_loss)
        actor_loss = -torch.sum(torch.stack(log_probs) * advantages.detach()) - entropy_weight * torch.sum(torch.stack(entropies))
        #calculate MSE between the critic's values and the returns
        critic_loss = nn.MSELoss()(torch.stack(values).squeeze(), returns) # use .stack().squeeze() for error
        
        #backprop
        total_loss = actor_loss + critic_loss
        total_loss.backward()
        
        actor_optimizer.step()
        critic_optimizer.step()

        rewards_per_episode.append(sum(rewards))
        loss_over_episodes.append(total_loss.item())

        print(f"Episode {episode+1}: Total reward = {sum(rewards)}, Entropy weight = {entropy_weight}")


    eps = range(num_episodes)
    plt.figure(figsize=(10, 5)) 
    plt.subplot(1, 2, 1) 
    plt.plot(eps, rewards_per_episode)

    plt.subplot(1, 2, 2)
    plt.plot(eps, loss_over_episodes)

    plt.tight_layout()
    plt.show()

env = gym.make("Acrobot-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

actor = Actor(state_dim, action_dim)
critic = Critic(state_dim)
actor_optimizer = optim.Adam(actor.parameters(), lr=0.01)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.01)


train(env, actor, critic, actor_optimizer, critic_optimizer, num_episodes=500)
