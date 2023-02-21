import os
import pandas as pd
import logging
from preprocess.dataset import Dataset

logging.basicConfig(level=logging.INFO, format=f'%(module)s/%(filename)s [Class: %(name)s Func: %(funcName)s] %(levelname)s : %(message)s')

class Stratify(Dataset):
    '''
    Stratification of Data: Number of Patients - Age, Dosage of Medication, Gender - Males (50-60), Based on diagnosis of Patient 
    Characteristics of Patients:
        Gender
        BMI
        Age group 
        Ethnicity
        Lengths of stay
        Mortality Rate
        Diagnosis
        Dosage of medication
    Otherss
    '''

    def __init__(self, name, data_path, types=None, preprocessed=False, n_sub=15000, random_seed=10, between_meds=(1,2), bmi=(20,25), age=(50,60), gender='both', ethnicity='WHITE'):
        
        self.DATA_P = os.path.join(data_path, 'preprocessed')
        self.DATA = data_path
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.types = types if types is not None else ['bmi', 'gender', 'age', 'ethnicity', 'length_stay', 'mortality']
        
        # Characteristics to stratify on
        self.bmi = bmi
        self.age = age
        self.gender = gender
        self.ethnicity = ethnicity

        self.logger.info(f'Started stratification')
        self.stratified = self.get_characteristics()

        Dataset.__init__(self, name, data_path, preprocessed, n_sub, random_seed, between_meds)
        self.logger.info(f'Completed stratification')
    
    def get_patient_data(self):

        ## Load admissions and patient data
        admissions = pd.read_csv(os.path.join(self.DATA, 'raw/ADMISSIONS.csv.gz'))
        patients = pd.read_csv(os.path.join(self.DATA, 'raw/PATIENTS.csv.gz'))

        # Convert to days
        patients['DOB'] = pd.to_datetime(patients['DOB']).dt.date
        admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME']).dt.date

        # Get first admission
        admissions = admissions.sort_values(['SUBJECT_ID', 'ADMITTIME']).groupby('SUBJECT_ID').nth(0).reset_index()

        # Merge admissions table and patients table
        temp = pd.merge(admissions, patients, how='inner', on='SUBJECT_ID')

        # Calculate age
        temp['AGE'] = temp.apply(lambda r: round((r['ADMITTIME']-r['DOB']).days/365, 0), axis=1)
        temp = temp[temp['AGE']<100]

        return temp

    def get_characteristics(self):

        patient_data = self.get_patient_data()

        bmi_data = pd.read_csv(os.path.join(self.DATA_P, 'bmi.csv')).drop(columns=['Unnamed: 0'])
        age = patient_data[['SUBJECT_ID', 'HADM_ID', 'AGE']]
        gender = patient_data[['SUBJECT_ID', 'HADM_ID', 'GENDER']]
        ethnicity = patient_data[['SUBJECT_ID', 'HADM_ID', 'ETHNICITY']]

        if 'bmi' in self.types:
            bmi_data = bmi_data[bmi_data['BMI']<=self.bmi[1]]
            bmi_data = bmi_data[bmi_data['BMI']>=self.bmi[0]]

        if 'age' in self.types:
            age = age[age['AGE']>=self.age[0]]
            age = age[age['AGE']<=self.age[1]]

        if 'gender' in self.types:
            if self.gender=='M':
                gender = gender[gender['GENDER']=='M']
            elif self.gender=='F':
                gender = gender[gender['GENDER']=='F']
            else:
                gender = gender[(gender['GENDER']=='F') | (gender['GENDER']=='M')]
        
        if 'ethnicity' in self.types:
            ethnicity = ethnicity[ethnicity['ETHNICITY']==self.ethnicity]
            
        final_data = pd.merge(age, pd.merge(bmi_data, gender, how='inner', on=['HADM_ID', 'SUBJECT_ID']), how='inner', on=['HADM_ID', 'SUBJECT_ID'])
        final_data = pd.merge(final_data, ethnicity, how='inner', on=['HADM_ID', 'SUBJECT_ID'])

        return final_data
    
    def preprocess(self, type='inputevents'):
        '''Data Preprocessing'''

        self.logger.info(f'Started Preprocessing {type} data....')

        lab_measurements = self.labevents

        if type=='inputevents':
            ### Patient Prescription
            patient_presc = self.inputevents

            # Get first admission
            # WRONG - patient_presc = self.remove_multiple_admissions(patient_presc)
            # patient_presc = patient_presc.sort_values(['SUBJECT_ID', 'STARTTIME']).groupby('SUBJECT_ID').nth(0).reset_index()
            patient_presc = patient_presc[patient_presc['HADM_ID'].isin(self.stratified['HADM_ID'])]
            patient_presc = patient_presc[patient_presc['LABEL'].isin(self.meds[type]['MED'])]
            patient_presc = patient_presc.sort_values(['HADM_ID', 'STARTTIME'])

            ### Lab Measurements
            lab_measurements = lab_measurements[lab_measurements['HADM_ID'].isin(patient_presc['HADM_ID'])]

            if self.between_meds:
                self.logger.info(f'Between {self.between_meds} medications')
                self.med2 = patient_presc.groupby(['HADM_ID', 'ITEMID']).nth(self.between_meds[1]).reset_index()
                self.med1 = patient_presc.groupby(['HADM_ID', 'ITEMID']).nth(self.between_meds[0]).reset_index()
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
