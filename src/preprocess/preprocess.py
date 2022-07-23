# Medication-Labtest Pairs Retieval and T-Test P-values
# Original file is located at
# Regression-Medication-Labtest_Pairs_Retrieval-5.ipynb
#     https://colab.research.google.com/drive/14H2102C8R0F2SWpoJewNwebX3wVkpfo_

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
import logging


class Dataset:
    
    def __init__(self, name, data_path, preprocessed=True, n_sub=15000, random_seed=10):
        self.name = name    # mimiciii
        self.DATA = data_path        
        self.logger = logging.getLogger(__name__)

        self.logger.info('Started loading data from ', name, ' dataset...')
        
        # Choose n random subjects
        self.dob_patient_bins = self.load_data('dob_patient_bins')
        random.seed(random_seed)
        subjects_2k = random.sample(list(self.dob_patient_bins['SUBJECT_ID'].value_counts().keys()), n_sub)
        
        # dataset data
        self.admissions = self.load_data('admissions')
        self.labevents = self.load_data('labevents')

        if preprocessed:
            self.inputevents = self.load_data('inputevents_mv_preprocessed')
        else:
            self.inputevents = self.load_data('inputevents')
        
        
        if preprocessed:
            self.prescriptions = self.load_data('prescription_preprocessed')
        else:
            self.prescriptions = self.load_data('prescriptions')

        self.logger.info('Loaded data from ', name, ' dataset.')
        self.logger.info('Preprocessing data from ', name, ' dataset...')
        
        self.patient_presc, self.lab_measurements = self.preprocess()

        self.logger.info('Preprocessed data from ', name, ' dataset.')
    
    def load_data(self, type):
        '''Load Data'''

        ### Admissions
        if type=='admissions':
            try:
                self.logger.info('Loading ', type, ' data...')
                admissions = pd.read_csv(os.path.join(self.DATA, 'ADMISSIONS.csv.gz'))
            except FileNotFoundError:
                self.logger.error('File not found Error in', type)
                return None
            else:
                self.logger.info('Loaded ', type)
                # subject_id,hadm_id
                admissions = admissions[['SUBJECT_ID', 'HADM_ID']]
                return admissions
        
        ### Labevents
        if type=='labevents':
            try:
                self.logger.info('Loading ', type, ' data...')
                labevents = pd.read_csv(os.path.join(self.DATA, 'LABEVENTS.csv.gz')).dropna()
                d_labitems = pd.read_csv(os.path.join(self.DATA, 'D_LABITEMS.csv.gz')).dropna()
                labValues = pd.merge(labevents, d_labitems, on='ITEMID', how='inner')
            except FileNotFoundError:
                self.logger.error('File not found Error in', type)
                return None
            else:
                self.logger.info('Loaded ', type)
                # subject_id,l.hadm_id, d.label, l.valuenum, l.valueuom, l.charttime
                labValues = labValues[['SUBJECT_ID', 'HADM_ID', 'LABEL', 'VALUENUM', 'VALUEUOM', 'CHARTTIME']]

                labValues['CHARTTIME'] = pd.to_datetime(labValues['CHARTTIME'],  format='%Y/%m/%d %H:%M:%S')
                return labValues

        ### Longitudanal Patient Data
        if type=='dob_patient_bins':
            try:
                self.logger.info('Loading ', type, ' data...')
                patients = pd.read_csv(os.path.join(DATA, 'PATIENTS.csv.gz'))
            except:
                self.logger.error('File not found Error ')
                return None
            else:
                self.logger.info('Loaded ', type)
                patients['DOB'] = pd.to_datetime(patients['DOB'],  format='%Y/%m/%d %H:%M:%S')
                k = patients.hist('DOB', bins=10, figsize=(12,8))

                bins = [np.array(patients['DOB'].min()).astype('datetime64'), np.datetime64('2050-01-01'), np.datetime64('2150-12-31'), np.array(patients['DOB'].max()).astype('datetime64')]
                temp1 = patients.copy()
                temp1['bins'] = pd.cut(temp1['DOB'], bins)

                temp1[temp1['bins']==temp1['bins'].unique()[0]]

                bins = [np.array(patients['DOB'].min()).astype('datetime64'), np.datetime64('2070-01-01'), np.datetime64('2150-12-31'), np.array(patients['DOB'].max()).astype('datetime64')]
                temp = patients.copy()
                temp['bins'] = pd.cut(temp['DOB'], bins)

                return temp[temp['bins']==temp['bins'].unique()[0]]

        ### Inputevents
        if type=='inputevents_mv':
            try:
                self.logger.info('Loading ', type, ' data...')

                inputevents_mv = pd.read_csv(os.path.join(self.DATA, 'INPUTEVENTS_MV.csv.gz'), nrows=10)
                with gzip.open(os.path.join(self.DATA, 'INPUTEVENTS_MV.csv.gz'), 'rb') as fp:
                    for i, k in enumerate(fp):
                        pass
                size = i+1
                # size of input events is 3618992

                data = []
                headers = None
                with gzip.open(os.path.join(self.DATA, 'INPUTEVENTS_MV.csv.gz'), 'rt') as fp:
                    reader = csv.reader(fp)
                    headers = next(reader)
                    for line in reader:
                        if int(line[1]) in subjects_2k and line[-6]=="FinishedRunning" and line[-7]=='0':
                            data.append(line)
                inputevents_mv_subjects = pd.DataFrame(data, columns=headers)

                d_item = pd.read_csv(os.path.join(self.DATA, 'D_ITEMS.csv.gz'))

            except FileNotFoundError:
                self.logger.error('File not found Error ')
                return None
            else:
                for i in ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ITEMID']:
                    print(i)
                    inputevents_mv_subjects[i] = inputevents_mv_subjects[i].astype('int64')

                ditem_inputevents_mv = pd.merge(inputevents_mv_subjects, d_item, on='ITEMID', how='inner')

                inputevents_mv_1 = ditem_inputevents_mv[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTTIME', 'ENDTIME', 'ITEMID', 'AMOUNT', 'AMOUNTUOM', 'UNITNAME', 'ORDERCATEGORYNAME', 'LABEL', 'CATEGORY', 'PARAM_TYPE']]
                inputevents_mv_1

                inputevents_mv_1['STARTTIME'] = pd.to_datetime(inputevents_mv_1['STARTTIME'],  format='%Y/%m/%d %H:%M:%S')
                inputevents_mv_1['ENDTIME'] = pd.to_datetime(inputevents_mv_1['ENDTIME'],  format='%Y/%m/%d %H:%M:%S')

                self.logger.info('Loaded ', type)

                return inputevents_mv_1
        
        ### Preprocessed Inputevents
        if type=='inputevents_mv_preprocessed':
            try:
                self.logger.info('Loading ', type, ' data...')
                inputevents_mv1 = pd.read_csv(os.path.join(RESULT, 'inputevents_mv_preprocessed.csv'))
            except FileNotFoundError:
                self.logger.error('File not found Error ')
                return None
            else:
                inputevents_mv1 = inputevents_mv1.drop(columns=['Unnamed: 0'])

                inputevents_mv1['STARTTIME'] = pd.to_datetime(inputevents_mv1['STARTTIME'],  format='%Y/%m/%d %H:%M:%S')
                inputevents_mv1['ENDTIME'] = pd.to_datetime(inputevents_mv1['ENDTIME'],  format='%Y/%m/%d %H:%M:%S')

                return inputevents_mv1

        ### Meds
        if type=='meds':
            try:
                self.logger.info('Loading ', type, ' data...')
                top200_meds = inputevents_mv_1['LABEL'].value_counts()[:200]
            except:
                self.logger.error('Error')
                return None
            else:
                self.logger.info('Loaded ', type)
                top200_meds = pd.DataFrame(top200_meds, columns=['LABEL']).reset_index()
                top200_meds.rename(columns = {'index':'MED', 'LABEL':'COUNT'}, inplace = True)
                return top200_meds
        
        if type=='prescriptions':
            
            try:
                with gzip.open(os.path.join(DATA, 'PRESCRIPTIONS.csv.gz'), 'rb') as fp:
                    for i, k in enumerate(fp):
                        pass
                size = i+1
                # size of input events is 4156451

                data = []
                headers = None
                with gzip.open(os.path.join(DATA, 'PRESCRIPTIONS.csv.gz'), 'rt') as fp:
                    reader = csv.reader(fp)
                    headers = next(reader)
                    for line in tqdm(reader, total=size):
                        if int(line[1]) in subjects_2k:
                            data.append(line)
                
                prescriptions = pd.DataFrame(data, columns=headers)
            except:
                self.logger.error('Error in ', type)
                return None
            else:
                prescriptions['STARTDATE'] = pd.to_datetime(prescriptions['STARTDATE'],  format='%Y/%m/%d %H:%M:%S')
                prescriptions['ENDDATE'] = pd.to_datetime(prescriptions['ENDDATE'],  format='%Y/%m/%d %H:%M:%S')

                for i in ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ITEMID']:
                    prescriptions[i] = prescriptions[i].astype('int64')

                self.logger.info('Loaded ', type)

                return prescriptions

        # preocessed prescription data
        if type=='prescription_preprocessed':
            try:
                self.logger.info('Loading ', type, ' data...')
                raise Exception('Pending Implementation')
            except:
                self.logger.error('Error')
                return None
            else:
                self.logger.info('Loaded ', type)
                return presc
        
        # preocessed prescription data
        if type=='prescription_preprocessed':
            try:
                self.logger.info('Loading ', type, ' data...')
                presc = pd.read_csv(os.path.join(RESULT, 'prescription_preprocessed.csv'))
            except:
                self.logger.error('Error')
                return None
            else:
                presc['STARTDATE'] = pd.to_datetime(presc['STARTDATE'],  format='%Y/%m/%d %H:%M:%S')
                presc['ENDDATE'] = pd.to_datetime(presc['ENDDATE'],  format='%Y/%m/%d %H:%M:%S')
                self.logger.info('Loaded ', type)
                return presc

    def remove_multiple_admissions(self, df):
        '''
        Removes hospital admissions that occur more than once for the same patient
    
        Parameters:
        df (DataFrame): Takes in dataframe with multiple hospital admissions
    
        Returns:
        Dataframe: Returns dataframe with multiple hospital admissions removed
        '''

        first_admissions = self.admissions
        first_admissions = first_admissions.drop_duplicates(subset=['SUBJECT_ID'], keep='first')
        df = df[df['HADM_ID'].isin(first_admissions['HADM_ID'])]
        return df

    def preprocess(self):
        '''Data Preprocessing'''

        self.logger.info('Started Preprocessing data....')

        lab_measurements = self.labevents

        ### Patient Prescription
        patient_presc = self.inputevents
        patient_presc = self.remove_multiple_admissions(patient_presc)
        patient_presc = self.inputevents[self.inputevents['LABEL'].isin(top200_meds['MED'])]

        ### Lab Measurements
        lab_measurements = lab_measurements[lab_measurements.duplicated(subset=['SUBJECT_ID','LABEL'],keep=False)]
        lab_measurements = lab_measurements[lab_measurements['HADM_ID'].isin(patient_presc['HADM_ID'])]

        self.logger.info('Processed data loaded to RAM.')

        return patient_presc, lab_measurements
