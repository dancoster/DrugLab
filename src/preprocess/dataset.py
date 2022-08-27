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
import random

logging.basicConfig(level=logging.INFO, format=f'%(module)s/%(filename)s [Class: %(name)s Func: %(funcName)s] %(levelname)s : %(message)s')

class Dataset:
    
    def __init__(self, name, data_path, preprocessed=True, n_sub=15000, random_seed=10, between_meds=(1,2)):
        self.name = name    # mimiciii
        self.DATA = data_path
        self.logger = logging.getLogger(self.__class__.__name__)

        self.between_meds = (between_meds[0]-1, between_meds[1]-1)
        if self.between_meds:
            self.med1, self.med2, self.only_med1 = None, None, None

        self.logger.info(f'Started loading data from {name} dataset...')
        
        # Choose n random subjects
        self.dob_patient_bins = self.load_data('dob_patient_bins')
        random.seed(random_seed)
        self.subjects_2k = random.sample(list(self.dob_patient_bins['SUBJECT_ID'].value_counts().keys()), n_sub)
        
        # dataset data
        self.admissions = self.load_data('admissions')
        self.labevents = self.load_data('labevents')
        self.meds = dict()

        if preprocessed:
            self.inputevents = self.load_data('inputevents_mv_preprocessed')
        else:
            self.inputevents = self.load_data('inputevents_mv')
        self.meds['inputevents'] = self.load_data('inputevent_meds')
        
        if preprocessed:
            self.prescriptions = self.load_data('prescription_preprocessed')
        else:
            self.prescriptions = self.load_data('prescriptions')
        self.meds['prescriptions'] = self.load_data('prescription_meds')

        self.logger.info(f'Loaded data from {name} dataset.')
        self.logger.info(f'Preprocessing data from {name} dataset...')

        self.patient_presc, self.lab_measurements = dict(), dict()
        self.patient_presc['inputevents'], self.lab_measurements['inputevents'] = self.preprocess('inputevents')
        self.patient_presc['prescriptions'], self.lab_measurements['prescriptions'] = self.preprocess('prescriptions')

        self.logger.info(f'Preprocessed data from {name} dataset.')
    
    def load_data(self, type):
        '''Load Data'''

        ### Admissions
        if type=='admissions':
            try:
                self.logger.info(f'Loading {type} data...')
                admissions = pd.read_csv(os.path.join(self.DATA, 'raw', 'ADMISSIONS.csv.gz'))
            except FileNotFoundError:
                self.logger.error(f'File not found Error in {type}')
                return None
            else:
                self.logger.info(f'Loaded {type}')
                # subject_id,hadm_id
                admissions = admissions[['SUBJECT_ID', 'HADM_ID']]
                return admissions
        
        ### Labevents
        if type=='labevents':
            try:
                self.logger.info(f'Loading {type} data...')
                labevents = pd.read_csv(os.path.join(self.DATA, 'raw', 'LABEVENTS.csv.gz')).dropna()
                d_labitems = pd.read_csv(os.path.join(self.DATA, 'raw', 'D_LABITEMS.csv.gz')).dropna()
                labValues = pd.merge(labevents, d_labitems, on='ITEMID', how='inner')
            except FileNotFoundError:
                self.logger.error(f'File not found Error in {type}')
                return None
            else:
                self.logger.info(f'Loaded {type}')
                # subject_id,l.hadm_id, d.label, l.valuenum, l.valueuom, l.charttime
                labValues = labValues[['SUBJECT_ID', 'HADM_ID', 'LABEL', 'VALUENUM', 'VALUEUOM', 'CHARTTIME']]

                labValues['CHARTTIME'] = pd.to_datetime(labValues['CHARTTIME'],  format='%Y/%m/%d %H:%M:%S')
                return labValues

        ### Longitudanal Patient Data
        if type=='dob_patient_bins':
            try:
                # print('Loading ', type, ' data...')
                self.logger.info(f'Loading {type} data...')
                patients = pd.read_csv(os.path.join(self.DATA, 'raw', 'PATIENTS.csv.gz'))
            except:
                self.logger.error(f'File not found Error ')
                return None
            else:
                self.logger.info(f'Loaded {type}')
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

        if type=='inputevents_mv':
            try:
                self.logger.info(f'Loading {type} data...')
                inputevents_mv_subjects = pd.read_csv(os.path.join(self.DATA, 'raw', 'INPUTEVENTS_MV.csv.gz'))
                inputevents_mv_subjects = inputevents_mv_subjects[inputevents_mv_subjects['STATUSDESCRIPTION']=='FinishedRunning']
                inputevents_mv_subjects = inputevents_mv_subjects[inputevents_mv_subjects['CANCELREASON']==0]
                d_item = pd.read_csv(os.path.join(self.DATA, 'raw', 'D_ITEMS.csv.gz'))          
            except:
                self.logger.error(f'File not found {type}')
                return None
            else:
                ditem_inputevents_mv = pd.merge(inputevents_mv_subjects, d_item, on='ITEMID', how='inner')
                inputevents_mv_1 = ditem_inputevents_mv[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTTIME', 'ENDTIME', 'ITEMID', 'AMOUNT', 'AMOUNTUOM', 'UNITNAME', 'ORDERCATEGORYNAME', 'LABEL', 'CATEGORY', 'PARAM_TYPE']]
                inputevents_mv_1['STARTTIME'] = pd.to_datetime(inputevents_mv_1['STARTTIME'],  format='%Y/%m/%d %H:%M:%S')
                inputevents_mv_1['ENDTIME'] = pd.to_datetime(inputevents_mv_1['ENDTIME'],  format='%Y/%m/%d %H:%M:%S')
                self.logger.info(f'Loaded {type}')
                return inputevents_mv_1                

        ### Inputevents
        if type=='inputevents_mv_low_mem':
            try:
                self.logger.info(f'Loading {type} data...')

                inputevents_mv = pd.read_csv(os.path.join(self.DATA, 'raw', 'INPUTEVENTS_MV.csv.gz'), nrows=10)
                with gzip.open(os.path.join(self.DATA, 'raw', 'INPUTEVENTS_MV.csv.gz'), 'rb') as fp:
                    for i, k in enumerate(fp):
                        pass
                size = i+1
                # size of input events is 3618992

                data = []
                headers = None
                with gzip.open(os.path.join(self.DATA, 'raw', 'INPUTEVENTS_MV.csv.gz'), 'rt') as fp:
                    reader = csv.reader(fp)
                    headers = next(reader)
                    for line in reader:
                        if int(line[1]) in self.subjects_2k and line[-6]=="FinishedRunning" and line[-7]=='0':
                            data.append(line)
                inputevents_mv_subjects = pd.DataFrame(data, columns=headers)

                d_item = pd.read_csv(os.path.join(self.DATA, 'raw', 'D_ITEMS.csv.gz'))

            except FileNotFoundError:
                self.logger.error(f'File not found Error ')
                return None
            else:
                for i in ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ITEMID']:
                    print(i)
                    inputevents_mv_subjects[i] = inputevents_mv_subjects[i].astype('int64')

                ditem_inputevents_mv = pd.merge(inputevents_mv_subjects, d_item, on='ITEMID', how='inner')

                inputevents_mv_1 = ditem_inputevents_mv[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTTIME', 'ENDTIME', 'ITEMID', 'AMOUNT', 'AMOUNTUOM', 'UNITNAME', 'ORDERCATEGORYNAME', 'LABEL', 'CATEGORY', 'PARAM_TYPE']]

                inputevents_mv_1['STARTTIME'] = pd.to_datetime(inputevents_mv_1['STARTTIME'],  format='%Y/%m/%d %H:%M:%S')
                inputevents_mv_1['ENDTIME'] = pd.to_datetime(inputevents_mv_1['ENDTIME'],  format='%Y/%m/%d %H:%M:%S')

                self.logger.info(f'Loaded {type}')

                return inputevents_mv_1
        
        ### Preprocessed Inputevents
        if type=='inputevents_mv_preprocessed':
            try:
                self.logger.info(f'Loading {type} data...')
                inputevents_mv1 = pd.read_csv(os.path.join(self.DATA, 'preprocessed',  'inputevents_mv_preprocessed.csv'))
            except FileNotFoundError:
                self.logger.error(f'File not found Error ')
                return None
            else:
                inputevents_mv1 = inputevents_mv1.drop(columns=['Unnamed: 0'])

                inputevents_mv1['STARTTIME'] = pd.to_datetime(inputevents_mv1['STARTTIME'],  format='%Y/%m/%d %H:%M:%S')
                inputevents_mv1['ENDTIME'] = pd.to_datetime(inputevents_mv1['ENDTIME'],  format='%Y/%m/%d %H:%M:%S')

                return inputevents_mv1

        ### Meds
        if type=='inputevent_meds':
            try:
                self.logger.info(f'Loading {type} data...')
                meds = self.inputevents.groupby(['LABEL', 'SUBJECT_ID']).count().reset_index()['LABEL'].value_counts().reset_index()
            except:
                self.logger.error(f'Error {type}')
                return None
            else:
                self.logger.info(f'Loaded {type}')
                meds.rename(columns = {'index':'MED', 'LABEL':'COUNT'}, inplace = True)
                return meds
        
        ### Meds
        if type=='prescription_meds':
            try:
                self.logger.info(f'Loading {type} data...')
                drugs = self.prescriptions.groupby(['DRUG', 'SUBJECT_ID']).count().reset_index()['DRUG'].value_counts().reset_index()

            except:
                self.logger.error(f'Error {type}')
                return None
            else:
                self.logger.info(f'Loaded {type}')
                drugs.rename(columns = {'index':'MED', 'DRUG':'COUNT'}, inplace = True)
                return drugs
        
        if type=='prescriptions':
            try:
                self.logger.info(f'Loading {type} data...')
                prescriptions = pd.read_csv(os.path.join(self.DATA, 'raw', 'PRESCRIPTIONS.csv.gz'))
            except:
                self.logger.error(f'File not found {type}.')
                return None
            else:
                self.logger.info(f'Loaded {type}')
                prescriptions['STARTDATE'] = pd.to_datetime(prescriptions['STARTDATE'],  format='%Y/%m/%d %H:%M:%S')
                prescriptions['ENDDATE'] = pd.to_datetime(prescriptions['ENDDATE'],  format='%Y/%m/%d %H:%M:%S')
                self.logger.info(f'Loaded {type}')
                return prescriptions
        
        if type=='prescriptions_low_mem':
            
            try:
                with gzip.open(os.path.join(self.DATA, 'raw', 'PRESCRIPTIONS.csv.gz'), 'rb') as fp:
                    for i, k in enumerate(fp):
                        pass
                size = i+1
                # size of input events is 4156451

                data = []
                headers = None
                with gzip.open(os.path.join(self.DATA, 'raw', 'PRESCRIPTIONS.csv.gz'), 'rt') as fp:
                    reader = csv.reader(fp)
                    headers = next(reader)
                    for line in tqdm(reader, total=size):
                        if int(line[1]) in self.subjects_2k:
                            data.append(line)
                
                prescriptions = pd.DataFrame(data, columns=headers)
            except:
                self.logger.error(f'Error in {type}')
                return None
            else:
                prescriptions['STARTDATE'] = pd.to_datetime(prescriptions['STARTDATE'],  format='%Y/%m/%d %H:%M:%S')
                prescriptions['ENDDATE'] = pd.to_datetime(prescriptions['ENDDATE'],  format='%Y/%m/%d %H:%M:%S')

                for i in ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ITEMID']:
                    prescriptions[i] = prescriptions[i].astype('int64')

                self.logger.info(f'Loaded {type}')

                return prescriptions

        # preocessed prescription data
        if type=='prescription_preprocessed':
            try:
                self.logger.info(f'Loading {type} data...')
                presc = pd.read_csv(os.path.join(self.DATA, 'preprocessed',  'prescription_preprocessed.csv'))
            except:
                self.logger.error(f'Error')
                return None
            else:
                presc['STARTDATE'] = pd.to_datetime(presc['STARTDATE'],  format='%Y/%m/%d %H:%M:%S')
                presc['ENDDATE'] = pd.to_datetime(presc['ENDDATE'],  format='%Y/%m/%d %H:%M:%S')
                self.logger.info(f'Loaded {type}')
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

    def get_meds(self, patient_presc):
        med2 = patient_presc.groupby('HADM_ID').nth(self.between_med[1]).reset_index()
        med1 = patient_presc.groupby('HADM_ID').nth(self.between_med[0]).reset_index()
        med1 = med1[med1['HADM_ID'].isin(med2['HADM_ID'])]        
        return med1, med2


    def preprocess(self, type='inputevents'):
        '''Data Preprocessing'''

        self.logger.info(f'Started Preprocessing {type} data....')

        lab_measurements = self.labevents

        if type=='inputevents':
            ### Patient Prescription
            patient_presc = self.inputevents

            # Get first admission
            # WRONG - patient_presc = self.remove_multiple_admissions(patient_presc)## Load admissions and patient data
            admissions = pd.read_csv(os.path.join(self.DATA, 'raw/ADMISSIONS.csv.gz'))
            admissions = admissions.sort_values(['SUBJECT_ID', 'ADMITTIME']).groupby('SUBJECT_ID').nth(0).reset_index() 

            patient_presc = patient_presc[patient_presc['HADM_ID'].isin(admissions['HADM_ID'])]
            patient_presc = patient_presc[patient_presc['LABEL'].isin(self.meds[type]['MED'])]

            ### Lab Measurements
            lab_measurements = lab_measurements[lab_measurements['HADM_ID'].isin(patient_presc['HADM_ID'])]

            if self.between_meds:
                self.med2 = patient_presc.groupby('HADM_ID').nth(self.between_meds[1]).reset_index()
                self.med1 = patient_presc.groupby('HADM_ID').nth(self.between_meds[0]).reset_index()
                self.only_med1 = self.med1[~self.med1['HADM_ID'].isin(self.med2['HADM_ID'])]
                self.med1 = self.med1[self.med1['HADM_ID'].isin(self.med2['HADM_ID'])]


        if type=='prescriptions':
            ### Patient Prescription
            patient_presc = self.prescriptions
            patient_presc = self.remove_multiple_admissions(patient_presc)
            patient_presc = self.prescriptions[self.prescriptions['DRUG'].isin(self.meds[type]['MED'])]

            ### Lab Measurements
            # lab_measurements = lab_measurements[lab_measurements.duplicated(subset=['SUBJECT_ID','LABEL'],keep=False)]
            lab_measurements = lab_measurements[lab_measurements['HADM_ID'].isin(patient_presc['HADM_ID'])]

        self.logger.info(f'Processed {type} data loaded to RAM.')

        return patient_presc, lab_measurements
