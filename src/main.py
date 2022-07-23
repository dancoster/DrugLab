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

def get_arg_val(arg, args):
    if arg in args:
        ind = args.index(arg)
        val = args[ind+1]
        return val
    return None

def main(args, BASE_DIR):
    print('Started process')

    logger = logging.getLogger(__name__)
    
    data_path = get_arg_val('--data', args)
    DATA = BASE_DIR+'/data'
    if data_path is None:
        data_path = os.path.join(DATA, 'mimiciii', '1.4')

    results_path = get_arg_val('--results', args)
    if results_path is None:
        results_path = BASE_DIR+'/results'

    n_meds = int(get_arg_val('--meds', args))
    if n_meds is None:
        n_meds = 50

    # Load dataset
    logger.info(f'Started loading dataset...')
    name = 'mimiciii'
    data_obj_path = os.path.join(data_path, 'preprocessed', name+'_dataset.obj')
    data = None
    if os.path.exists(data_obj_path):
        logger.info(f'Preprocessed dataset found. Loading dataset object...')
        file = open(data_obj_path, 'rb')
        data = pickle.load(file)
        file.close()
    else:
        logger.info(f'Preprocessed dataset not found. Preprocessing and Loading dataset')
        data = Dataset(name, data_path)
        file = open(data_obj_path, 'wb')
        pickle.dump(data, file)
        file.close()
    logger.info(f'Done loading dataset.')

    # Analysis
    # logger.info(f'Started inputevents data analysis...')
    # inp_path = os.path.join(results_path, 'inputevents')
    # inputevents_analysis = IEDataAnalysis(inp_path, data)
    # inputevents_analysis.analyse(n_meds=n_meds)
    # logger.info(f'Done analyzing inputevents data. Results can be found in {inp_path} directory.')

    # Analysis
    # logger.info(f'Started prescriptions data analysis...')
    # pres_path = os.path.join(results_path, 'prescriptions')
    # prescriptions_analysis = PRDataAnalysis(pres_path, data)
    # prescriptions_analysis.analyse(n_meds=n_meds)
    # logger.info(f'Done analyzing prescriptions data. Results can be found in {pres_path} directory.')

    # Visualize
    logger.info(f'Started visualizing results of data analysis...')
    plot_module = TimeEffectVisualization('Insulin - Regular', 'Glucose', BASE_DIR, data)
    plot_module.visualize()
    logger.info(f'Done visualizing.')
    

if __name__=="__main__":    
    BASE_DIR = os.path.dirname(os.getcwd())
    main(sys.argv, BASE_DIR)