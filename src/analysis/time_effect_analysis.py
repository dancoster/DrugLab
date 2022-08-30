# spearman's rho to get a correlation coefficient for each participant (I only have 7 time points so I guess that's why it's spearman??). Then enter all of the coefficients in a single sample t-test, testing it against the value of zero. 
# Pearsons COefficient
# 

import pandas as pd
import datetime
import random
import numpy as np

from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr, spearmanr
from scipy import stats

from tqdm import tqdm
import os
import gzip
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics
import logging

from analysis.analysis import Analysis

logging.basicConfig(level=logging.INFO, format=f'%(module)s/%(filename)s [Class: %(name)s Func: %(funcName)s] %(levelname)s : %(message)s')

class TimeEffect(Analysis):

    def __init__(self, path, dataset, table):
        self.data = dataset

        self.patient_presc = None
        self.lab_measurements = None
        self.meds = None

        if table=='inputevents':
            self.patient_presc = self.data.patient_presc[table]
            self.lab_measurements = self.data.lab_measurements[table]
            self.meds = self.data.meds[table][:20]
        elif table=='prescriptions':
            self.patient_presc = self.data.patient_presc[table]
            self.lab_measurements = self.data.lab_measurements[table]
            self.meds = self.data.meds[table][:20]
        
        self.logger = logging.getLogger(self.__class__.__name__)
        Analysis.__init__(self, path, dataset, table)
    
    def correlations_analysis(self, presc, lab, window=(1,24), before_window=None, after_windows=None, val_type='absolute'):
        p_corrs, s_corrs, time = list(), list(), list()
        for after_window in after_windows:
            p_corr, s_corr = self.get_correlation(presc, lab, before_window=before_window, after_window=after_window, val_type=val_type, window=window)
            p_corrs.append(p_corr)
            s_corrs.append(s_corr)
        return p_corrs, s_corrs

    def get_correlation(self, presc, lab, window=(1,24), before_window=None, after_window=None, corr_type=None, val_type='absolute', method='estimate'):

        values, time_diff = self.get_data(presc, lab, val_type, method=method, before_window=before_window, after_window=after_window, window=window)

        if corr_type=='pearson':
            '''
            Linear relation between values and time
            '''
            corr, _ = pearsonr(values, time_diff)
            self.logger.info('Pearsons correlation: %.3f' % corr)

        if corr_type=='spearmans':
            '''
            Non Linear relation
            '''
            corr, _ = spearmanr(values, time_diff)
            self.logger.info('Spearmans correlation: %.3f' % corr)
        
        if corr_type is None:
            p_corr, _ = pearsonr(values, time_diff)
            s_corr, _ = spearmanr(values, time_diff)
            return p_corr, s_corr
                
        return corr
    
    def get_data(self, presc, lab, type, method='estimate', before_window=None, after_window=None, window=(1,24)):
        '''
        Data Collection and processing
        '''

        # 'Insulin - Regular'
        # 'Glucose'
        drug_lab, before1, after1 = Analysis.labpairing(presc, self.patient_presc, self.lab_measurements, lab, type=self.table, med1=self.data.med1, med2=self.data.med2, window=window)

        self.logger.info(f'Before Subjects: , {drug_lab}, {after1}, {before1}')

        subjects = list(drug_lab['SUBJECT_ID'].unique())

        self.logger.info(f'Data: , {after1}, {before1}')
        self.logger.info(f'Params : , {after_window}, {before_window}')
        
        if method=='before-after':            
            if after_window is not None:
                after1 = after1[(
                        (
                            after1['timeFromPrescription']>datetime.timedelta(hours=after_window[0])
                        ) & (
                            after1['timeFromPrescription']<datetime.timedelta(hours=after_window[1])
                        )
                    )]                   
            if self.table=='inputevents':
                before1['timeFromPrescription'] = before1['timeFromPrescription'].apply(lambda x : round(x.total_seconds()/3600, 2) )
                before1 = before1.sort_values(by='timeFromPrescription')
                before1 = before1.groupby('SUBJECT_ID').last().reset_index()['VALUENUM']      

        if before_window is not None:
            before1 = before1[(
                    (
                        before1['timeFromPrescription']<datetime.timedelta(hours=(-1*before_window[0]))
                    ) & (
                        before1['timeFromPrescription']>datetime.timedelta(hours=(-1*before_window[1]))
                    )
                )]
            after1 = after1[after1['SUBJECT_ID'].isin(before1['SUBJECT_ID'])]
            before1 = before1[before1['SUBJECT_ID'].isin(after1['SUBJECT_ID'])]

        self.logger.info(f'{after1}, {before1}')

        reg_anal_res, _, _ = Analysis.interpolation(subjects, before1)
        if method=='estimate':
            e = pd.DataFrame(reg_anal_res)
            self.logger.info(f'{e}')
            e = e.rename(columns={'subjectID':'SUBJECT_ID'})
            estimate = e['estimated']
        if method=='before-after':
            estimate = before1.rename(columns={'VALUENUM':'estimated'})['estimated']

        if self.table=='inputevents':
            after1['timeFromPrescription'] = after1['timeFromPrescription'].apply(lambda x : round(x.total_seconds()/3600, 2) )
            after = after1.groupby('SUBJECT_ID').first().reset_index()['VALUENUM']
            time_diff = after1.groupby('SUBJECT_ID').first().reset_index()['timeFromPrescription'] 
        if self.table=='prescriptions':                
            merged = pd.merge(drug_lab, e, how='inner', on='SUBJECT_ID').rename(columns={'timeFromPrescription_y':'timeFromPrescription', 'VALUENUM_y':'VALUENUM'})
            time_diff = merged['timeFromPrescription']
            time_diff = time_diff.apply(lambda t : t.total_seconds()/3600)
            after = merged['VALUENUM']
                
        if type=='absolute':
            absolute = after-estimate
            return absolute, time_diff
        elif type=='percent':
            percent = 100*(after-estimate)/estimate
            return percent, time_diff
        elif type=='ratio':
            ratio = after/estimate
            return ratio, time_diff
        else:
            self.logger.error('Choose correct type')
            return

    def remove_outlier(self, val, time_diff):
        val = pd.DataFrame(val)
        time_diff = pd.DataFrame(time_diff)
        
        # IQR
        Q1 = np.percentile(val, 25, interpolation = 'midpoint')        
        Q3 = np.percentile(val, 75, interpolation = 'midpoint')
        IQR = Q3 - Q1        
        
        # Upper bound
        upper = np.where(val >= (Q3+1.5*IQR))
        # Lower bound
        lower = np.where(val <= (Q1-1.5*IQR))

        # Filtering
        val.drop(upper[0], inplace = True)
        time_diff.drop(upper[0], inplace = True)
        val.drop(lower[0], inplace = True)
        time_diff.drop(lower[0], inplace = True)
        return val, time_diff
