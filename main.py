import argparse, os
from ac_acrobat import hyperparameter_search, regularization_search, variance_study
import create_graphs 


def main():
    parser = argparse.ArgumentParser(description="Policy based RL")
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--regularization-study", action="store_true")
    parser.add_argument("--variance-study", action="store_true")
    parser.add_argument("--proces-results", action="store_true")

    
    args = parser.parse_args()

    print("possible commands:")
    print("--grid-search: initialize randomized grid searched over \
hyperparameter space")
    print("--regularization-study: train agents with different amount \
        of entropy regularization to study effect")
    print("--variance-study: runs the algorithm with four different configurations \
and saves the average of 5 runs to json files")
    print("--proces-results: proces results of grid search and \
studies to generate graphs and tables used in the report")
    input("Press enter to continue...")

    if args.grid_search:
        grid_search = True
    else:
        grid_search = False

    if args.regularization_study:
        regularization_study = True 
    else:
        regularization_study = False  
    
    if args.variance_study:
        variance_stud = True
    else: 
        variance_stud = False

    if args.proces_results:
        proces_results = True
    else:
        proces_results = False
    
    # gridsearch hyperparameter space
    search_space = {
    'hidden_nodes': [16, 32, 64, 128],
    'eta': [1, 0.1, 0.05 ,0.01, 0.005],
    'lr': [0.0001, 0.001, 0.01],
    'baseline': [True,False],
    'bootstrap':[True,False],
    'n_step': [3, 5, 7, 10],
    'gamma': [.9,0.99,.999]
    }     


    # create folder for saving results
    if not os.path.exists('results'):
        os.makedirs('results')
    
    if grid_search:
        hyperparameter_search(200,save_path = 'results')

    if regularization_study:
        regularization_search()
    
    if variance_stud:
        variance_study()
        variance_study(save_path = 'results')

    if proces_results:
        create_graphs.main()

if __name__ == "__main__":
    main()
