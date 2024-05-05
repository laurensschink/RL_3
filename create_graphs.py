import pickle, os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from itertools import product

def smooth(y, window, poly=2):
    #Helper function to smooth loss functions
    return savgol_filter(y, window, poly)

def load_results(result_path = 'results'):
    with open('RL_3/results/grid_search_ac3.pickle','rb') as f:
        results_ac = pickle.load(f)
    with open('RL_3/results/vary_eta.pickle', 'rb') as f:
        eta_dict = pickle.load(f)

    df_ac = pd.DataFrame().from_dict(results_ac)
    df_ac[['expansions']] = 0
    df_ac.loc[df_ac.baseline, 'expansions'] += 1 
    df_ac.loc[df_ac.bootstrap, 'expansions'] += 2 
    df_ac['label'] = df_ac['expansions'].map({0:'none',1:'baseline',2:'bootstrap',3:'both'})
    return results_ac,eta_dict
   
def main():
    results_ac,eta_dict = load_results()

    # create table with top perfroming algorithm configurations
    report_cols = ['hidden_nodes', 'eta', 'lr', 'label', 'n_step', 'gamma','avg_reward', 'avg_eval_reward']
    df_ac[report_cols].sort_values('avg_eval_reward',ascending=False).head(10).to_csv('table_top_10_params.csv')

    # Learning curves for best performing algorithms
    # indices in 'best' dictionary hand picked
    best = {'REINFOCRCE': 177,
        'bootstrap': 189 ,
        'baseline': 42,
        'both': 77}
    colors = ['mediumorchid','lightblue','lightcoral','lightgreen','lightsalmon','lightyellow']
    for i,(key,idx) in enumerate(best.items()):
        train = df_ac.iloc[idx]['learning_curve']
        
        eval_dict = df_ac.iloc[idx]['eval_curve']
        x = eval_dict.keys()
        y = [eval_dict[k] for k in x]
        plt.plot(smooth(train,9), label=f'train {key}', linestyle = '--', color = colors[i])
        plt.plot(x,y, color = colors[i], label = f'eval {key}')
    plt.xlim(0,500)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Learning curves for the best performing configuration')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',title="Rewards per AC option")
    plt.savefig(os.path.join('results','learning_curves.png'))
    plt.show()

    # grid search results per parameter
    params = ['hidden_nodes', 'eta', 'lr', 'expansions', 'n_step', 'gamma']
    fig,axs = plt.subplots(2,3,figsize=(8,6), sharey=True)
    for i,(param,ax) in enumerate(zip(params,axs.ravel())):

        xy = np.vstack([df_ac[param],df_ac['avg_eval_reward']])
        z = gaussian_kde(xy)(xy)
        scatter =  ax.scatter(df_ac[param],df_ac['avg_eval_reward'],c=z,s=4)
        ax.set_title(param)
        if i%3==0:
            ax.set_ylabel('Average reward')
        if param == 'expansions':
            ax.set_xticks(range(4),['REINFORCE', 'baseline', 'bootstrap', 'both'], rotation=45)
        if param in ['eta','lr']:
            ax.set_xscale('log')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  
    plt.suptitle(f'Grid search Actor Critic after {len(df_ac)} searches')
    plt.savefig(os.path.join('results','grid_search_graphs.png'))
    plt.show()

    # Regularization plot
    fig, axs = plt.subplots(2,2, figsize=(10,8), sharex=True,sharey=True)
    for base,boot in product([False,True],[False,True]):

        for i,eta in enumerate(etas):
            train_rewards = np.array(eta_dict[(base,boot)][eta]['rewards'])
            mean_train = train_rewards.mean(axis=0)
            std_train = train_rewards.std(axis=0)
            
            eval_rewards = np.array([[reward[key] for reward in eta_dict[(base,boot)][eta]['eval_rewards']] for key in keys]).T
            mean_eval = eval_rewards.mean(axis=0)
            std_eval = eval_rewards.std(axis=0)

            axs[base*1,boot*1].plot(smooth(mean_train,9),label = f'train reward, eta: {eta}', linestyle='--',color=colors[i])
    #        plt.fill_between(np.arange(len(mean_train)),smooth(mean_train,9)-smooth(std_train,9), smooth(mean_train,9)+smooth(std_train,9),alpha=.3)
            axs[base*1,boot*1].plot(keys,mean_eval,label = f'eval reward, eta: {eta}',color=colors[i] )
            axs[base*1,boot*1].fill_between(keys,mean_eval-std_eval,mean_eval+std_eval,alpha=.3,color=colors[i] , label = 'error')
        axs[base*1,boot*1].set_ylim(-600,-100)
        title = 'REINFORCE' 
        if base:
            title = 'Actor critic: baseline'
        if boot:
            title = 'Actor critic: bootstrap'
        if base & boot: 
            title = 'Actor critic: baseline & bootstrap' 
        axs[base*1,boot*1].set_title(title,fontsize=16)
        if not boot:
            axs[base*1,boot*1].set_ylabel('Reward',fontsize=12)
        if base:
            axs[base*1,boot*1].set_xlabel('Episode',fontsize=12)

    axs[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left',frameon=True)
    plt.suptitle('Effect of entropy regularization', fontsize=20)
    plt.savefig(os.path.join('results','regularization.png'))
    plt.show()

if __name__ == '__main__':
    main()