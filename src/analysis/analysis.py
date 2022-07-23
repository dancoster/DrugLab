import pandas as pd
import datetime
import random
import numpy as np
from scipy.stats import mannwhitneyu
from scipy import stats
from tqdm import tqdm
import os
import gzip
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics

class Analysis:

    def __init__(self, path, dataset, type):
        self.RESULTS = path
        self.data = dataset
        self.type = type

    def analyse(self, n_subs=200, n_meds=50):

        table = self.type

        patient_presc = self.data.patient_presc[table]
        lab_measurements = self.data.lab_measurements[table]
        meds = self.data.meds[table]

        ## Generating Lab Test<>Meds Pairings
        finalDF, before, after = Analysis.labpairing('NaCl 0.9%', patient_presc, lab_measurements, 'Calcium, Total', type=table)

        ## Final Results - Reading before and after, regression and trend
        res = self.results_analysis(patient_presc, lab_measurements, meds, n_subs, n_meds)
        time = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
        res.to_csv(os.path.join(self.RESULTS, f'{table}_before_after_interpolation_trend_{time}.csv'))

    def results_generator(self, med, presc, lab_measurements, labTest, n_medlab_pairs):
        pass

    def results_analysis(self, lab_measurements,  patient_presc, meds, n_medlab_pairs, n_meds):
        pass

    @staticmethod
    def labpairing(medname, prescdf, labdf, labname, k=3, type='inputevents'):
        '''
        Generating Lab Test<>Meds Pairings. Pairs the drug input with each lab test

        Parameters:
        drugname (String): Drug Name
        prescdf (DataFrame): Dataframe containing the prescription data
        labdf (DataFrame): Dataframe containing the lab measurement data
        labname (DataFrame): Lab Test Name
        Returns:
        DataFrame: Contains all the rows of values and times for that particular drug lab apir
        '''

        if type=='inputevents':
            
            # Select patients who have taken the drug
            prescdf = prescdf[prescdf['LABEL']==medname]
            prescdf = prescdf.drop_duplicates(subset=['SUBJECT_ID'], keep='first')

            # Select lab measurements of patients who have taken the drug
            labdf = labdf[labdf['HADM_ID'].isin(prescdf['HADM_ID'])]

            # Selects the lab measurement entered
            drug_lab_specific = labdf[labdf['LABEL']==labname]
            mergeddf = pd.merge(drug_lab_specific, prescdf, on=['HADM_ID','SUBJECT_ID'])

            # Get time from prescription and choose before and after lab measurements (within k days)
            mergeddf['timeFromPrescription'] = mergeddf['CHARTTIME'] - mergeddf['STARTTIME']
            mergeddf = mergeddf[(mergeddf['timeFromPrescription']>datetime.timedelta(days=(-1*k))) & (mergeddf['timeFromPrescription']<datetime.timedelta(days=k))]
            posmergeddf = mergeddf.loc[mergeddf.timeFromPrescription > datetime.timedelta(hours=0)]
            negmergeddf = mergeddf.loc[mergeddf.timeFromPrescription < datetime.timedelta(hours=0)]
            
            # Only keep values for which we have both before and after
            posmergeddf = posmergeddf[posmergeddf['HADM_ID'].isin(negmergeddf['HADM_ID'])]
            negmergeddf = negmergeddf[negmergeddf['HADM_ID'].isin(posmergeddf['HADM_ID'])]
            drug_lab = negmergeddf.merge(posmergeddf,on=['HADM_ID','SUBJECT_ID'])

            # Choose admissions which have more than one lab test reading
            before = negmergeddf
            bool_before = before.groupby('SUBJECT_ID').count()>1
            index_before = bool_before[bool_before['HADM_ID']==True].index
            before = before[before['SUBJECT_ID'].isin(index_before)]
            negmergeddf = negmergeddf.loc[negmergeddf.groupby('SUBJECT_ID').timeFromPrescription.idxmax()]

            after = posmergeddf
            bool_after = after.groupby('SUBJECT_ID').count()>1
            index_after = bool_after[bool_after['HADM_ID']==True].index
            after = after[after['SUBJECT_ID'].isin(index_after)]
            posmergeddf = posmergeddf.loc[posmergeddf.groupby('SUBJECT_ID').timeFromPrescription.idxmin()]

            before = before[before['HADM_ID'].isin(after['HADM_ID'])]
            after = after[after['HADM_ID'].isin(before['HADM_ID'])]

            finaldf = negmergeddf.merge(posmergeddf,on=['HADM_ID','SUBJECT_ID'])
            
            return finaldf, before, after
        
        elif type=='prescriptions':

            # Select patients who have taken the drug
            prescdf = prescdf[prescdf['DRUG']==medname]
            prescdf = prescdf.drop_duplicates(subset=['SUBJECT_ID'], keep='first')

            # Select lab measurements of patients who have taken the drug
            labdf = labdf[labdf['HADM_ID'].isin(prescdf['HADM_ID'])]

            # Selects the lab measurement entered
            drug_lab_specific = labdf[labdf['LABEL']==labname]
            mergeddf = pd.merge(drug_lab_specific, prescdf, on=['HADM_ID','SUBJECT_ID'])

            # Get time from prescription and choose before and after lab measurements (within 24hrs=1day)
            mergeddf['timeFromPrescription'] = mergeddf['CHARTTIME'] - mergeddf['STARTDATE']
            mergeddf = mergeddf[(mergeddf['timeFromPrescription']>datetime.timedelta(days=(-1*k))) & (mergeddf['timeFromPrescription']<datetime.timedelta(days=k))]
            posmergeddf = mergeddf.loc[mergeddf.timeFromPrescription > datetime.timedelta(hours=12)]
            negmergeddf = mergeddf.loc[mergeddf.timeFromPrescription < datetime.timedelta(hours=-12)]
            
            # Only keep values for which we have both before and after
            posmergeddf = posmergeddf[posmergeddf['HADM_ID'].isin(negmergeddf['HADM_ID'])]
            negmergeddf = negmergeddf[negmergeddf['HADM_ID'].isin(posmergeddf['HADM_ID'])]
            df = negmergeddf.merge(posmergeddf,on=['HADM_ID','SUBJECT_ID'])

            # Choose admissions which have more than one lab test reading
            before = negmergeddf
            bool_before = before.groupby('SUBJECT_ID').count()>1
            index_before = bool_before[bool_before['HADM_ID']==True].index
            before = before[before['SUBJECT_ID'].isin(index_before)]
            negmergeddf = negmergeddf.loc[negmergeddf.groupby('SUBJECT_ID').timeFromPrescription.idxmax()]

            after = posmergeddf
            bool_after = after.groupby('SUBJECT_ID').count()>1
            index_after = bool_after[bool_after['HADM_ID']==True].index
            after = after[after['SUBJECT_ID'].isin(index_after)]
            posmergeddf = posmergeddf.loc[posmergeddf.groupby('SUBJECT_ID').timeFromPrescription.idxmin()]

            before = before[before['HADM_ID'].isin(after['HADM_ID'])]
            after = after[after['HADM_ID'].isin(before['HADM_ID'])]

            finaldf = negmergeddf.merge(posmergeddf,on=['HADM_ID','SUBJECT_ID'])
            
            return finaldf, before, after

    ## Regression-Trend Analysis
    @staticmethod
    def get_min(col):
        return col.apply(lambda td : td.total_seconds()//60)

    @staticmethod
    def get_hour(col):
        return col.apply(lambda td : td.total_seconds()//3600)

    # Regression Analysis
    @staticmethod
    def interpolation(subjects, before, plot=False):
        reg_anal_res = []
        lab_vals = []
        time = []

        for i in subjects:
            
            res_vals = dict()
            res_vals['subjectID'] = i
            
            rows = before[before['SUBJECT_ID']==i]
            rows = rows.sort_values(by='timeFromPrescription')

            x = rows['VALUENUM']

            if plot:
                y = Analysis.get_hour(rows['timeFromPrescription'])
            else:
                y = Analysis.get_min(rows['timeFromPrescription'])

            reg = linear_model.LinearRegression()
            if np.array(x).shape[0]>0 and np.array(y).shape[0]>0:
                reg = linear_model.LinearRegression()
                reg.fit(np.array(y).reshape(-1,1), np.array(x).reshape(-1,1))

                res_vals['coef'] = reg.coef_[0][0]
                res_vals['estimated'] = reg.predict([[0]])[0][0]

                reg_anal_res.append(res_vals)

                lab_vals.append(x)
                time.append(y)

        return reg_anal_res, lab_vals, time
