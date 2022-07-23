import os, sys
from preprocess.preprocess import Dataset
from analysis.inputevents_analysis import InputeventsDataAnalysis
from analysis.prescriptions_analysis import PrescriptionsDataAnalysis
from visualize.lab_time_difference import TimeEffectVisualization
import pandas as pd
import datetime
import random
import numpy as np
from scipy.stats import mannwhitneyu
from scipy import stats
import os
import gzip
import csv
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics
import seaborn as sns
import logging
import pickle

logging.basicConfig(level=logging.INFO, format=f'%(filename)s [Class: %(name)s Func:%(funcName)s] %(levelname)s : %(message)s')

def main(args, BASE_DIR):
    print('Started process')

    data_path = None
    logger = logging.getLogger(__name__)
    
    if '--data' in args:
        ind = args.index('--data')
        data_path = args[ind+1]
    else:
        DATA = BASE_DIR+'/data'
        data_path = os.path.join(DATA, 'mimiciii', '1.4')
    
    results_path = None
    if '--results' in args:
        ind = args.index('--results')
        results_path = args[ind+1]
    else:
        results_path = BASE_DIR+'/results'

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
    logger.info(f'Started inputevents data analysis...')
    inp_path = os.path.join(results_path, 'inputevents')
    inputevents_analysis = InputeventsDataAnalysis(inp_path)
    inputevents_analysis.analyse()
    logger.info(f'Done analyzing inputevents data. Results can be found in {inp_path} directory.')

    # Analysis
    logger.info(f'Started prescriptions data analysis...')
    pres_path = os.path.join(results_path, 'prescriptions')
    prescriptions_analysis = PrescriptionsDataAnalysis(pres_path)
    prescriptions_analysis.analyse()
    logger.info(f'Done analyzing prescriptions data. Results can be found in {pres_path} directory.')

    # Visualize
    

if __name__=="__main__":    
    BASE_DIR = os.path.dirname(os.getcwd())
    main(sys.argv, BASE_DIR)