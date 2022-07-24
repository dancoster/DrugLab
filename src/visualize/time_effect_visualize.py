import pandas as pd
import datetime
import random
import numpy as np

from scipy.stats import mannwhitneyu
from scipy import stats

import os
import gzip
import csv
from sklearn import datasets, linear_model, metrics

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns


import logging

from analysis.analysis import Analysis
from analysis.time_effect_analysis import TimeEffect

logging.basicConfig(level=logging.INFO, format=f'%(module)s/%(filename)s [Class: %(name)s Func: %(funcName)s] %(levelname)s : %(message)s')

class TimeEffectVisualization(TimeEffect):
    '''
    Lab VS Time difference Plot
    '''

    def __init__(self, BASE_DIR, dataset, table='inputevents'):
        '''
        Inputevents and prescription table
        '''       
        self.logger = logging.getLogger(self.__class__.__name__) 
        self.BASE_DIR = BASE_DIR
        self.plots_path = os.path.join(self.BASE_DIR, 'plots')     
        
        TimeEffect.__init__(self, self.plots_path, dataset, table)


    def visualize(self, presc, lab, path=None, window=(1,24)):
        '''
        Visualize and plotting
        '''
        
        comp_path = self.plots_path
        if path is not None:
            comp_path = path
        
        absolute, time_diff = self.get_data( presc, lab, 'absolute')
        percent, time_diff = self.get_data( presc, lab, 'percent')
        ratio, time_diff = self.get_data( presc, lab, 'ratio')
        
        # suffix = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
        suffix = f'w{str(window[0])}-{str(window[1])}'
        dirname = lab+"<>"+presc+"_"+suffix
        dirpath = os.path.join(self.BASE_DIR, 'plots', 'inputevents', dirname)
        if not os.path.isdir(dirpath):
            os.mkdir(dirpath)

        corrs = self.get_correlation(presc, lab)
        absolute1, time_diff3 = self.remove_outlier(absolute, time_diff)
        self.plot_func(presc, lab, absolute1, time_diff3, dirname, window=window, title='Absolute', labels=corrs)

        corrs = self.get_correlation(presc, lab, val_type='percent')
        percent1, time_diff1 = self.remove_outlier(percent, time_diff)
        self.plot_func(presc, lab, percent1, time_diff1, dirname, window=window, title='Percentage', labels=corrs)

        corrs = self.get_correlation(presc, lab, val_type='ratio')
        ratio1, time_diff2 = self.remove_outlier(ratio, time_diff)
        self.plot_func(presc, lab, ratio1, time_diff2, dirname, window=window, title='Ratio', labels=corrs)

    def plot_func(self, presc, lab, absolute, time_diff, dirname, window=(1,24), title='', unit='mg/dL', labels=None):
        plot_data = pd.concat([absolute, time_diff], axis=1)
        plot_data = plot_data.rename(columns={0:'Lab values'})

        plot_data = plot_data[plot_data['timeFromPrescription'] > window[0]]
        plot_data = plot_data[plot_data['timeFromPrescription'] < window[1]]

        sns.regplot(x = "timeFromPrescription", 
                y = 'Lab values', 
                data = plot_data, 
                truncate=False)
        plt.title(lab+'<>'+presc+'- '+ title+ ' \nchange in lab measurment and time taken for change')
        plt.xlabel('Time in hours')
        plt.ylabel(lab+' Levels ('+unit+')')
        if labels is not None:
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            plt.legend([extra, extra], (f'Pearson Correlation = {round(labels[0], 2)}', f'Spearmans Correlation = {round(labels[1], 2)}'))
        plt.savefig(os.path.join(self.BASE_DIR, 'plots', 'inputevents', dirname, title+".png"))
        plt.clf()

# ## Prescriptions

# merged = pd.merge(drug_lab, e, how='inner', on='SUBJECT_ID')
# absolute = merged['VALUENUM_y']-e['estimated']

# time_diff = merged['timeFromPrescription_y']

# time_diff = time_diff.apply(lambda t : t.total_seconds()/3600)

# after = merged['VALUENUM_y']
# estimate = e['estimated']

# percent = 100*(after-estimate)/estimate

# ratio = after/estimate

# def remove_outlier(val, time_diff):
#     val = pd.DataFrame(val)
#     time_diff = pd.DataFrame(time_diff)
#     # IQR
#     Q1 = np.percentile(val, 25,
#                     interpolation = 'midpoint')
    
#     Q3 = np.percentile(val, 75,
#                     interpolation = 'midpoint')
#     IQR = Q3 - Q1
    
#     # Upper bound
#     upper = np.where(val >= (Q3+1.5*IQR))
#     # Lower bound
#     lower = np.where(val <= (Q1-1.5*IQR))
#     val.drop(upper[0], inplace = True)
#     time_diff.drop(upper[0], inplace = True)
#     val.drop(lower[0], inplace = True)
#     time_diff.drop(lower[0], inplace = True)
#     return val, time_diff

# import seaborn as sns

# def plot_func(absolute, time_diff, title=''):
#     plot_data = pd.concat([absolute, time_diff], axis=1)
#     plot_data = plot_data.rename(columns={0:'Lab values'})
#     plot_data = plot_data[plot_data['timeFromPrescription_y']>12]
#     sns.regplot(x = "timeFromPrescription_y", 
#             y = 'Lab values', 
#             data = plot_data, 
#             truncate=False)
#     plt.title('Insulin<>Glucose - '+ title+ ' change in lab measurment and time taken for change')
#     plt.xlabel('Time in hours')
#     plt.ylabel('Glucose Levels (mg/dL)')
#     plt.show()