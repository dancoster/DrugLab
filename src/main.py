import os, sys
from preprocess.preprocess import Dataset
from analysis.inputevents_analysis import IEDataAnalysis
from analysis.prescriptions_analysis import PRDataAnalysis
from visualize.time_effect_visualize import TimeEffectVisualization

import pandas as pd
import datetime
import random
import numpy as np
from scipy.stats import mannwhitneyu
from scipy import stats
import gzip
import csv
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics
import seaborn as sns
import logging
import pickle
from scipy.stats import mannwhitneyu
from scipy import stats

logging.basicConfig(level=logging.INFO, format=f'%(filename)s [Class: %(name)s Func:%(funcName)s] %(levelname)s : %(message)s')

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

    return config

def get_arg_val(arg, args, k=1):
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

    # Load dataset
    logger.info(f'Started loading dataset...')
    name = 'mimiciii'
    data_obj_path = os.path.join(config['data_path'], 'preprocessed', name+'_dataset.obj')
    data = None
    if os.path.exists(data_obj_path):
        logger.info(f'Preprocessed dataset found. Loading dataset object...')
        file = open(data_obj_path, 'rb')
        data = pickle.load(file)
        file.close()
    else:
        logger.info(f'Preprocessed dataset not found. Preprocessing and Loading dataset')
        data = Dataset(name, config['data_path'])
        file = open(data_obj_path, 'wb')
        pickle.dump(data, file)
        file.close()
    logger.info(f'Done loading dataset.')

    # # Analysis
    if '-i' in args:
        logger.info(f'Started inputevents data analysis...')
        inp_path = os.path.join(config['results_path'], 'inputevents')
        inputevents_analysis = IEDataAnalysis(inp_path, data)
        inputevents_analysis.analyse(n_meds=config['n_meds'], n_subs=config['n_sub_pairs'], window=config['window'])
        logger.info(f'Done analyzing inputevents data. Results can be found in {inp_path} directory.')

    # # Analysis
    if '-p' in args:
        logger.info(f'Started prescriptions data analysis...')
        pres_path = os.path.join(config['results_path'], 'prescriptions')
        prescriptions_analysis = PRDataAnalysis(pres_path, data)
        prescriptions_analysis.analyse(n_meds=config['n_meds'], n_subs=config['n_sub_pairs'], window=config['window'])
        logger.info(f'Done analyzing prescriptions data. Results can be found in {pres_path} directory.')

    # Visualize
    if '-v' in args:
        pair = config['visualize']
        logger.info(f'Started visualizing time effect analysis {str(pair)} of pair results...')
        plot_module = TimeEffectVisualization(BASE_DIR, data, table=config['table'])
        plot_module.visualize(config['visualize'][0], config['visualize'][1], window=config['window'])
        logger.info(f'Done visualizing.')
    

if __name__=="__main__":
    BASE_DIR = os.getcwd()
    main(sys.argv, BASE_DIR)