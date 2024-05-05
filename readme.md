# Reinforcement Learning Assignment 3
## By Bram van Eerden, Laurens Schinkelhoek, and Jelmer Stroo

### Code structure:
main.py for running experiments 
reinforce_agent for the reinforce agent
ac_acrobat for the actor critic agent where bootstrapping & base line subtraction should be optional
create_graphs for processing results of eta/grid search

### Run instructions:
To run initialize main.py.  
This comes with the following options:  
--grid-search: initialize randomized grid search over hyperparameter space.  
--regularization-study: train agents with different amount of entropy regularization to study effect 
--variance-study: runs the algorithm with four different configurations (bootstrapping/baseline subtraction) and saves the average of 5 runs to json files
--proces-results: proces results of grid search and studies to generate graphs and tables used in the report

(Intermediate) results are saved in a folder 'results' that is created if it does not yet exist.

### Required packages:
torch, numpy, scipy, matplotlib, gymnasium, pandas

