import os, sys
from preprocess.preprocess import Dataset
from analysis.inputevents_analysis import IEDataAnalysis
from analysis.prescriptions_analysis import PRDataAnalysis
from visualize.lab_time_difference import TimeEffectVisualization

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

def config_args(args, BASE_DIR. DATA):

    config = dict()

    config['data_path'] = get_arg_val('--data', args)
    if config['data_path'] is None:
        config['data_path'] = os.path.join(DATA, 'mimiciii', '1.4')

    config['results_path'] = get_arg_val('--results', args)
    if config['results_path'] is None:
        config['results_path'] = BASE_DIR+'/results'

    config['n_meds'] = int(get_arg_val('--meds', args))
    if config['n_meds'] is None:
        config['n_meds'] = 50

    config['window'] = get_arg_val('--config['window']', args, k=2)
    if config['window'] is None:
        config['window'] = (1,24)
    else:
        config['window'] = (int(config['window'][0]), int(config['window'][1]))

    return config

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
    # logger.info(f'Started inputevents data analysis...')
    # inp_path = os.path.join(config['results_path'], 'inputevents')
    # inputevents_analysis = IEDataAnalysis(inp_path, data)
    # inputevents_analysis.analyse(config['n_meds']=config['n_meds'])
    # logger.info(f'Done analyzing inputevents data. Results can be found in {inp_path} directory.')

    # # Analysis
    # logger.info(f'Started prescriptions data analysis...')
    # pres_path = os.path.join(config['results_path'], 'prescriptions')
    # prescriptions_analysis = PRDataAnalysis(pres_path, data)
    # prescriptions_analysis.analyse(config['n_meds']=config['n_meds'])
    # logger.info(f'Done analyzing prescriptions data. Results can be found in {pres_path} directory.')

    # Visualize
    logger.info(f'Started visualizing results of data analysis...')
    plot_module = TimeEffectVisualization('Insulin - Regular', 'Glucose', BASE_DIR, data)
    plot_module.visualize(config['window']=config['window'])
    logger.info(f'Done visualizing.')
    

if __name__=="__main__":    
    BASE_DIR = os.path.dirname(os.getcwd())
    main(sys.argv, BASE_DIR)