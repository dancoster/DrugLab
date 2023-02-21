import os
import pandas as pd
import numpy as np
import sys
import logging

from analysis.significant import SignificantPairs


def config_args(args, BASE_DIR, DATA):

    config = dict()

    config['data_path'] = get_arg_val('--data', args)
    if config['data_path'] is None:
        config['data_path'] = os.path.join(DATA, 'mimiciii', '1.4')

    config['results_path'] = get_arg_val('--results', args)
    if config['results_path'] is None:
        config['results_path'] = BASE_DIR+'/results'

    config['n_meds'] = get_arg_val('--meds', args)
    if config['n_meds'] is None:
        config['n_meds'] = 50
    config['n_meds'] = int(config['n_meds'])

    config['n_sub_pairs'] = get_arg_val('--sub-pairs', args)
    if config['n_sub_pairs'] is None:
        config['n_sub_pairs'] = 50
    config['n_sub_pairs'] = int(config['n_sub_pairs'])

    config['table'] = get_arg_val('--table', args)
    if config['table'] is None:
        config['table'] = 'inputevents'

    config['window'] = get_arg_val('--window', args, k=2)
    if config['window'] is None:
        config['window'] = (1,24)
    else:
        config['window'] = (int(config['window'][0]), int(config['window'][1]))
    
    config['visualize'] = get_arg_val('--visualize', args, k=2)
    if config['visualize'] is None:
        config['visualize'] = ('Insulin - Regular', 'Glucose')
    else:
        config['visualize'] = tuple(config['visualize'])

    config['between_meds'] = get_arg_val('--between-meds', args, k=2)
    if config['between_meds'] is None:
        config['between_meds'] = (1, 2)
    else:
        config['between_meds'] = tuple([int(i) for i in config['between_meds']])
    
    config['before_window'] = get_arg_val('--before-window', args, k=2)
    if config['before_window'] is None:
        config['before_window'] = (2, 4)
    else:
        config['before_window'] = tuple([int(i) for i in config['before_window']])
    
    config['after_window'] = get_arg_val('--after-window', args, k=2)
    if config['after_window'] is None:
        config['after_window'] = (2, 5)
    else:
        config['after_window'] = tuple([int(i) for i in config['after_window']])

    config['stratify'] = get_arg_val('--stratify', args, 0)
    if config['stratify']:

        config['age'] = get_arg_val('-a', args, 2)
        if config['age'] is None:
            config['age'] = (50, 60)
        else:
            config['age'] = tuple([int(i) for i in config['age']])
        
        config['ethnicity'] = get_arg_val('-e', args)
        if config['ethnicity'] is None:
            config['ethnicity'] = 'WHITE'
        
        config['bmi'] = get_arg_val('-b', args, 2)
        if config['bmi'] is None:
            config['bmi'] = (18.4, 24.9)
        else:
            config['bmi'] = tuple([float(i) for i in config['bmi']])
        
        config['gender'] = get_arg_val('-g', args)
        if config['gender'] is None:
            config['gender'] = 'M'

    return config

def get_arg_val(arg, args, k=1):
    if k==0:
        if arg in args:
            return True
        else:
            return False
    if k==1:
        if arg in args:
            ind = args.index(arg)
            val = args[ind+1]
            return val
    if k>1:
        if arg in args:
            ind = args.index(arg)
            val = []
            for i in range(k):
                val.append(args[ind+i+1])
            return val
    return None

def main(args, BASE_DIR):
    print('Started process')
    DATA = BASE_DIR+'/data'

    logger = logging.getLogger(__name__)

    config = config_args(args, BASE_DIR, DATA)

    # suffix used by significant pair
    suffix1 = f'{config["between_meds"]}_{config["bmi"]}_{config["gender"]}_{config["age"]}_{config["ethnicity"]}'

    # res_path generation
    suffix = f'p{config["n_sub_pairs"]}_m{config["n_meds"]}_w{config["window"][0]}-{config["window"][1]}_{suffix1}'
    res_path = os.path.join('../results/Round-Results-Store', f'{config["table"]}_before_after_interpolation_trend_{suffix}.csv')

    
    significant = SignificantPairs('mannwhitney', suffix1)
    significant.get_intersection(res_path=res_path)
    

if __name__=="__main__":
    BASE_DIR = os.getcwd()
    main(sys.argv, BASE_DIR)