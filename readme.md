# Reinforcement Learning Assignment 3
## By Bram van Eerden, Laurens Schinkelhoek, and Jelmer Stroo

### Code structure:
Let's aim for this structure:
main.py for running experiments 
REINFORCE for the reinforce agent
actorcritic for the actor critic agent where bootstrapping & base line subtraction should be optional

Main.py initialises the environment and imports the agent relevant for each experiment.

### Run instructions:
To run initialize main.py.  
This comes with the following options:  
--grid-search: initialize randomized grid searched over hyperparameter space.  
--ablation-study-egreedy: run ablation study on optimal parameters for epsilon greedy decay exploration strategy.  
--ablation-study-botlzmann: run ablation study on optimal parameters for boltzmann exploration strategy.  
--proces-results: proces results of grid search and ablation study to generate graphs and tables used in the report.  

main.py must be run with one of these options.  
(Intermediate) results are saved in a folder 'results' that is created if it does not yet exist.

### Required packages:
torch, numpy, scipy, matplotlib, gymnasium, pandas

