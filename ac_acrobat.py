import gymnasium as gym
import numpy as np
import torch, os, pickle
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib
import matplotlib.pyplot as plt



#### Search parameters:
# basline, bootstrap, eta, lr
# hidden layer,
# n_step
# optimizer?

# simple, seperate actor-critic nns
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hiddden_nodes=32):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hiddden_nodes),
            nn.ReLU(),
            nn.Linear(hiddden_nodes, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state):
        action_probs = self.network(state)
        return action_probs

class Critic(nn.Module):
    def __init__(self, state_dim, hiddden_nodes=32):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hiddden_nodes),
            nn.ReLU(),
            nn.Linear(hiddden_nodes, 1)
        )
        
    def forward(self, state):
        value = self.network(state)
        return value

# new entropy calculation 
def compute_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-6), dim=1)

# also seperate optimizers for both nns
def train(env, actor, critic, actor_optimizer, critic_optimizer, 
        baseline = False, bootstrap = False, 
        initial_entropy_weight=0.01, n_step=5, gamma=0.99, num_episodes=2000,eval_rate = 100):

    # track gradients
    actor_grad_log = []
    critic_grad_log = []
    
    eval_rate = 100
    num_episodes = num_episodes
    final_entropy_weight = initial_entropy_weight/100
    loss_over_episodes = []
    rewards_per_episode = []
    eval_rewards = {}
    for episode in range(num_episodes):
        
        if episode%eval_rate == 0: # evaluation run
            eval_list = []
            for i in range(10):
                eval_reward = 0
                state, _ = env.reset()
                while True:
                    state = torch.from_numpy(state).float().unsqueeze(0)
                    probs = actor(state)
                    action = torch.argmax(probs).item()
                    state, reward, done, truncated, _ = env.step(action)
                    eval_reward+= reward
                    if done or truncated:
                        eval_list.append(eval_reward)
                        break
                eval_rewards[episode] = np.mean(eval_list)
        # should find out if this is enough learning episodes, and if 0.1 is high enough
        entropy_weight = max(initial_entropy_weight - (initial_entropy_weight - final_entropy_weight) * (episode / 200), final_entropy_weight) # decrease to 0.001 in the first 100 epsiodes

        # init the environment, init lists
        state,_ = env.reset()
        state = torch.from_numpy(state).float().unsqueeze(0)
        log_probs = []
        values = []
        rewards = []
        entropies = []
        states = [state]
        
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
            next_state, reward, done, truncated, _ = env.step(action.item())
            next_state = torch.from_numpy(next_state).float().unsqueeze(0)
            
            # get critic estimate for reward for current state
            value = critic(state)
            
            #store data
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)
            states.append(next_state)

            state = next_state
            if done or truncated:
                break
            
        #returns and advantages
        returns = []
        advantages = []
        G = 0

        # reversely calculate returns and advantages from the episode
        if bootstrap: # added bootstrap
            for t,reward in enumerate(rewards):
                n_step_t = min(n_step,len(rewards)-t)
                if t+n_step_t == len(rewards):
                    value = 0
                else:
                    value = critic(states[t+n_step_t])
                n_state_value = (gamma**n_step) * value 
                G = reward + torch.stack([(gamma**n) * torch.tensor(rewards[t+n]) for n in range(n_step_t)]).sum() + n_state_value
                returns.append(G)
                advantages.append(G - values[t].item())
            returns = torch.tensor(returns, dtype=torch.float32)
            advantages = torch.tensor(advantages, dtype=torch.float32)

        else:
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
        if baseline:
            actor_loss = -torch.sum(torch.stack(log_probs) * advantages) - entropy_weight * torch.sum(torch.stack(entropies))
        else:
            actor_loss = -torch.sum(torch.stack(log_probs) * returns) - entropy_weight * torch.sum(torch.stack(entropies))
        
        #calculate MSE between the critic's values and the returns
        critic_loss = nn.MSELoss()(torch.stack(values).squeeze(), returns) # use .stack().squeeze() for error

        #backprop
        total_loss = actor_loss + critic_loss
        total_loss.backward(retain_graph=True)
        
        actor_grads = []
        critic_grads = []
        
        for param in actor.parameters():
            if param.grad is not None:
                actor_grads.append(param.grad.view(-1).detach().cpu().numpy()**2)
        actor_grad_log.append(np.concatenate(actor_grads))
        
        for param in critic.parameters():
            if param.grad is not None:
                critic_grads.append(param.grad.view(-1).detach().cpu().numpy()**2)
        critic_grad_log.append(np.concatenate(critic_grads))    
        

        actor_optimizer.step()
        critic_optimizer.step()

        rewards_per_episode.append(np.sum(rewards))
        loss_over_episodes.append(total_loss.item())
    return actor_grad_log, critic_grad_log, rewards_per_episode, loss_over_episodes, eval_rewards


def experiment(hidden_nodes, eta, lr, baseline, bootstrap, n_step,gamma,max_eps=801):
    env = gym.make("Acrobot-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    train_rewards, train_losses, eval_rewards = list(),list(),list()
    for i in range(10):
        actor = Actor(state_dim, action_dim,hidden_nodes)
        critic = Critic(state_dim, hidden_nodes)
        actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
        critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
        train_score, train_loss, eval_score= train(env, actor, critic, actor_optimizer, critic_optimizer, baseline, bootstrap,
                    initial_entropy_weight=eta, n_step=n_step,  gamma=gamma, num_episodes=max_eps,eval_rate=25)
        train_rewards.append(train_score)
        train_losses.append(train_loss)
        eval_rewards.append(eval_score)
    return {'train_rewards':train_rewards,'train_losses':train_losses,'eval_rewards':eval_rewards}


def hyper_search_ac3(nr_searches = 100):
    env = gym.make("Acrobot-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    max_eps = 1501

    search_space = {
    'hidden_nodes': [16, 32, 64, 128],
    'eta': [1, 0.1, 0.05 ,0.01, 0.005],
    'lr': [0.0001, 0.001, 0.01],
    'baseline': [True,False],
    'bootstrap':[True,False],
    'n_step': [3, 5, 7, 10],
    'gamma': [0.99,0.9,1]
    }       

    summaries = []
    
    for c,search in enumerate(range(nr_searches)):
        param_dict = {key:np.random.choice(value) for key,value in search_space.items()}

        hidden_nodes = int(param_dict['hidden_nodes'])
        eta = param_dict['eta']
        lr = param_dict['lr']
        baseline = param_dict['baseline']
        bootstrap = param_dict['bootstrap']
        n_step = param_dict['n_step']
        gamma = param_dict['gamma']
        print(param_dict)

        actor = Actor(state_dim, action_dim,hidden_nodes)
        critic = Critic(state_dim, hidden_nodes)
        actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
        critic_optimizer = optim.Adam(critic.parameters(), lr=lr)

        episode_rewards, losses, eval_rewards= train(env, actor, critic, actor_optimizer, critic_optimizer, baseline, bootstrap,
                initial_entropy_weight=eta, n_step=n_step,  gamma=gamma,  num_episodes=max_eps)

        param_dict['avg_reward'] = np.mean(episode_rewards)
        param_dict['avg_eval_reward'] = np.mean(list(eval_rewards.values()))
        param_dict['learning_curve'] = episode_rewards
        param_dict['eval_curve'] = eval_rewards
        summaries.append(param_dict)
        print(f'run {c} - {nr_searches} completed.')
        print(f'mean reward: {param_dict["avg_reward"]:.2f}, mean eval: {param_dict["avg_eval_reward"]:.2f}')
        with open(os.path.join('results','grid_search_ac3.pickle'), 'wb') as f:
            pickle.dump(summaries, f)

    
def plot_gradient_variance(grad_logs, title="gradient variance for actor & critic"):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # plot on two axes
    color = 'tab:blue'
    ax1.set_xlabel('episode')
    ax1.set_ylabel('actor variance', color=color)
    ax1.plot([np.var(g) for g in grad_logs['Actor']], color=color)
    ax1.tick_params(axis='y', labelcolor=color)


    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel('critic variance', color=color)
    ax2.plot([np.var(g) for g in grad_logs['Critic']], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(title)
    fig.tight_layout()
    plt.show()

def main():

    # hyper_search_ac3(100)
    # experiment()
    
    env = gym.make("Acrobot-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)
    actor_optimizer = optim.AdamW(actor.parameters(), lr=0.001)
    critic_optimizer = optim.AdamW(critic.parameters(), lr=0.001)
    
    actor_grad_log, critic_grad_log, _, _, _ = train(env, actor, critic, actor_optimizer, critic_optimizer,False,False)
    # print ("actor_grads ", actor_grad_log)
    # print("critic_grads ", critic_grad_log)
    combined_logs = {
        "Actor": actor_grad_log,
        "Critic": critic_grad_log
    }
    
    plot_gradient_variance(combined_logs)

if __name__ == '__main__':
    main()