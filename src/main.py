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

logging.basicConfig(level=logging.INFO, format=f'%(asctime)s %(module)s/%(filename)s [Class: %(name)s Func:%(funcName)s] %(levelname)s : %(message)s')

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

    # Analysis
    inputevents_analysis = InputeventsDataAnalysis(results_path)
    inputevents_analysis.analyse()

    # Analysis
    prescriptions_analysis = PrescriptionsDataAnalysis(results_path)
    prescriptions_analysis.analyse()

    # Visualize
    

if __name__=="__main__":    
    
    BASE_DIR = os.path.dirname(os.getcwd())

    main(sys.argv, BASE_DIR)