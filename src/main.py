import os, sys
from preprocess.preprocess import Dataset

def main(args, BASE_DIR):
    print('Started process')

    data_path = None
    
    if '--data' in args:
        ind = args.index('--data')
        data_path = args[ind+1]
    else:
        DATA = BASE_DIR+'/data'
        data_path = os.path.join(DATA, 'mimiciii')
    
    results_path = None
    if '--results' in args:
        ind = args.index('--results')
        results_path = args[ind+1]
    else:
        results_path = BASE_DIR+'/results'

    # Load dataset
    data = Dataset('mimiciii', data_path)

if __name__=="__main__":    
    
    BASE_DIR = os.path.dirname(os.getcwd())

    main(sys.argv, BASE_DIR)