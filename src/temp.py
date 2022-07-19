# Medication-Labtest Pairs Retieval and T-Test P-values
# Original file is located at
# Regression-Medication-Labtest_Pairs_Retrieval-5.ipynb
#     https://colab.research.google.com/drive/14H2102C8R0F2SWpoJewNwebX3wVkpfo_

import pandas as pd
import datetime
import random
import numpy as np
# from scipy.stats import mannwhitneyu
# from scipy import stats
# from tqdm import tqdm
import os
import gzip
import csv
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from sklearn import datasets, linear_model, metrics

## Load Drive

PARENT='/content/drive/MyDrive/TAU'
DATA = PARENT+'/Datasets/mimiciii/1.4'
RESULT = PARENT+'/Results'

class Dataset:
    
    def __init__(self, name, data_path, preprocessed_inputevents=True):
        self.name = name    # mimiciii
        self.DATA = data_path

        # tables data
        self.admissions = self.load_data('admissions')
        self.labevents = self.load_data('labevents')
        self.dob_patient_bins = self.load_data('dob_patient_bins')
        if preprocessed_inputevents:
            self.inputevents = self.load_data('preprocessed_inputevents')
        else:
            self.inputevents = self.load_data('inputevents')
    
    def load_data(self, type):
        '''Load Data'''

        ### Admissions
        if type=='admissions':
            try:
                print('Loading ', type, ' data...')
                admissions = pd.read_csv(os.path.join(DATA, 'ADMISSIONS.csv.gz'))
            except FileNotFoundError:
                print('Error ', FileNotFoundError)
                return None
            else:
                print('Loaded ', type)
                # subject_id,hadm_id
                admissions = admissions[['SUBJECT_ID', 'HADM_ID']]
                return admissions
        
        ### Labevents
        if type=='labevents':
            try:
                print('Loading ', type, ' data...')
                labevents = pd.read_csv(os.path.join(DATA, 'LABEVENTS.csv.gz')).dropna()
                d_labitems = pd.read_csv(os.path.join(DATA, 'D_LABITEMS.csv.gz')).dropna()
                labValues = pd.merge(labevents, d_labitems, on='ITEMID', how='inner')
            except FileNotFoundError:
                print('Error ', FileNotFoundError)
                return None
            else:
                print('Loaded ', type)
                # subject_id,l.hadm_id, d.label, l.valuenum, l.valueuom, l.charttime
                labValues = labValues[['SUBJECT_ID', 'HADM_ID', 'LABEL', 'VALUENUM', 'VALUEUOM', 'CHARTTIME']]

                labValues['CHARTTIME'] = pd.to_datetime(labValues['CHARTTIME'],  format='%Y/%m/%d %H:%M:%S')
                return labValues

        ### Longitudanal Patient Data
        if type=='dob_patient_bins':
            try:
                print('Loading ', type, ' data...')
                patients = pd.read_csv(os.path.join(DATA, 'PATIENTS.csv.gz'))
            except FileNotFoundError:
                print('Error ', FileNotFoundError)
                return None
            else:
                print('Loaded ', type)
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
                print('Loading ', type, ' data...')
                random.seed(10)
                subjects_2k = random.sample(list(temp['SUBJECT_ID'].value_counts().keys()), 15000)

                inputevents_mv = pd.read_csv(os.path.join(DATA, 'INPUTEVENTS_MV.csv.gz'), nrows=10)

                with gzip.open(os.path.join(DATA, 'INPUTEVENTS_MV.csv.gz'), 'rb') as fp:
                    for i, k in enumerate(fp):
                        pass
                size = i+1
                # size of input events is 3618992

                data = []
                headers = None
                with gzip.open(os.path.join(DATA, 'INPUTEVENTS_MV.csv.gz'), 'rt') as fp:
                    reader = csv.reader(fp)
                    headers = next(reader)
                    for line in reader:
                        if int(line[1]) in subjects_2k and line[-6]=="FinishedRunning" and line[-7]=='0':
                            data.append(line)

                inputevents_mv_subjects = pd.DataFrame(data, columns=headers)

                d_item = pd.read_csv(os.path.join(DATA, 'D_ITEMS.csv.gz'))

            except FileNotFoundError:
                print('Error ', FileNotFoundError)
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

                print('Loaded ', type)

                return inputevents_mv_1
        
        ### Preprocessed Inputevents
        if type=='preprocessed_inputevents':
            try:
                print('Loading ', type, ' data...')
                inputevents_mv1 = pd.read_csv(os.path.join(RESULT, 'inputevents_mv_preprocessed.csv'))
            except FileNotFoundError:
                print('Error ', FileNotFoundError)
                return None
            else:
                print('Loaded ', type)
                inputevents_mv1 = inputevents_mv1.drop(columns=['Unnamed: 0'])

                inputevents_mv1['STARTTIME'] = pd.to_datetime(inputevents_mv1['STARTTIME'],  format='%Y/%m/%d %H:%M:%S')
                inputevents_mv1['ENDTIME'] = pd.to_datetime(inputevents_mv1['ENDTIME'],  format='%Y/%m/%d %H:%M:%S')

                return inputevents_mv1

        ### Meds
        if type=='meds':
            try:
                print('Loading ', type, ' data...')
                top200_meds = inputevents_mv_1['LABEL'].value_counts()[:200]
            except:
                print('Error')
                return None
            else:
                print('Loaded ', type)
                top200_meds = pd.DataFrame(top200_meds, columns=['LABEL']).reset_index()
                top200_meds.rename(columns = {'index':'MED', 'LABEL':'COUNT'}, inplace = True)
                return top200_meds

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

        lab_measurements = self.labevents

        ### Patient Prescription

        patient_presc = inputevents_mv_1

        patient_presc = remove_multiple_admissions(patient_presc)
        patient_presc = inputevents_mv_1[inputevents_mv_1['LABEL'].isin(top200_meds['MED'])]

        patient_presc

        ### Lab Measurements

        lab_measurements = lab_measurements[lab_measurements.duplicated(subset=['SUBJECT_ID','LABEL'],keep=False)]

        lab_measurements = lab_measurements[lab_measurements['HADM_ID'].isin(patient_presc['HADM_ID'])]

        lab_measurements

class DataAnalysis:

    def __init__(self, path):
        self.RESULTS = path

    def run(self):    

        ## Generating Lab Test<>Meds Pairings
        finalDF, before, after = labpairing('NaCl 0.9%', patient_presc, lab_measurements, 'Calcium, Total')

        ## Final Results - Reading before and after
        res = self.comp_analysis(lab_measurements, top200_meds[:60], 200)
        res.to_csv(os.path.join(self.RESULTS, 'trial.csv'))

            
        final_res_df, res_after_values = analysis_func(lab_measurements, top200_meds[:60], n_medlab_pairs = 200)
        final_res_df.to_csv(os.path.join(self.RESULTS, 'trial-2.csv'))



    ## Generating Lab Test<>Meds Pairings
    def labpairing(self, medname, prescdf, labdf, labname, k=3):
        '''Pairs the drug input with each lab test

        Parameters:
        drugname (String): Drug Name
        prescdf (DataFrame): Dataframe containing the prescription data
        labdf (DataFrame): Dataframe containing the lab measurement data
        labname (DataFrame): Lab Test Name
        Returns:
        DataFrame: Contains all the rows of values and times for that particular drug lab apir
        '''
        
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

    ## Final Results - Reading before and after
    def postprocessing(self, df):
        '''
        Gets the mean, standard deviation, mann whitney and t-test p values. Converts time delta to hours
    
        Parameters:
        df (DataFrame): Dataframe containing before and after lab test values and time values
        Returns:
        List:Containing mean, standard deviation, mann whitney and t-test p values and count
        '''
    
        
        df['timeFromPrescription_x'] = pd.to_numeric(df['timeFromPrescription_x'].dt.seconds)
        df['timeFromPrescription_x']/=3600
        df['timeFromPrescription_y'] = pd.to_numeric(df['timeFromPrescription_y'].dt.seconds)
        df['timeFromPrescription_y']/=3600
        df_before_mean = df['VALUENUM_x'].mean()
        df_after_mean = df['VALUENUM_y'].mean()
        df_before_std = df['VALUENUM_x'].std()
        df_after_std = df['VALUENUM_y'].std()
        df_before_time_mean = df['timeFromPrescription_x'].mean()
        df_after_time_mean = df['timeFromPrescription_y'].mean()
        df_before_time_std = df['timeFromPrescription_x'].std()
        df_after_time_std = df['timeFromPrescription_y'].std()
        ttestpvalue = stats.ttest_ind(df['VALUENUM_x'], df['VALUENUM_y'])[1]
        mannwhitneyu = stats.mannwhitneyu(df['VALUENUM_x'], df['VALUENUM_y'])[1]
        lengthofdf = len(df)
        csvrow=[lengthofdf,df_before_mean,df_before_std,df_before_time_mean,df_before_time_std,df_after_mean,df_after_std,df_after_time_mean,df_after_time_std,ttestpvalue, mannwhitneyu]
        return csvrow

    def comp_analysis(self, lab_measurements, top100_drugs, n_druglab_pairs = 25, n_drugs=None):
        res = pd.DataFrame(columns=['Medication','Lab Test','Number of patients','Lab Test Before(mean)','Lab Test Before(std)','Time Before(mean)','Time Before(std)','Lab Test After(mean)','Lab Test After(std)','Time After(mean)','Time After(std)', 'Ttest-pvalue', 'Mannwhitney-pvalue'])
        uniqueLabTests = lab_measurements.LABEL.unique()

        for i, drug in enumerate(top100_drugs['MED']): 
            if n_drugs is not None and i>=n_drugs:
                break
            print(i, ' Medication: ', drug)
            for j in range(uniqueLabTests.shape[0]):
                labTest = uniqueLabTests[j]
                drug_lab, after1, before1 = labpairing(drug, patient_presc, lab_measurements, labTest)
                if(len(drug_lab) > n_druglab_pairs): 
                    csvrow=self.postprocessing(drug_lab)
                    csvrow.insert(0, drug) 
                    csvrow.insert(1, labTest)
                    res.loc[len(res)] = csvrow
        return res

    ## Regression-Trend Analysis
    def get_min(col):
        return col.apply(lambda td : td.total_seconds()//60)

    def get_hour(col):
        return col.apply(lambda td : td.total_seconds()//3600)

    # Regression Analysis
    def gen_estimate_coef(subjects, before, verbose=False):
        reg_anal_res = []
        lab_vals = []
        time = []

        for i in subjects:
            
            res_vals = dict()
            res_vals['subjectID'] = i
            
            rows = before[before['SUBJECT_ID']==i]
            rows = rows.sort_values(by='timeFromPrescription')

            x = rows['VALUENUM']
            y = get_min(rows['timeFromPrescription'])

            reg = linear_model.LinearRegression()
            reg.fit(np.array(y).reshape(-1,1), np.array(x).reshape(-1,1))

            res_vals['coef'] = reg.coef_[0][0]
            res_vals['estimated'] = reg.predict([[0]])[0][0]
            
            if verbose:
                print('Subject ID: ', i, '  Coefficients: ', reg.coef_)

            reg_anal_res.append(res_vals)

            lab_vals.append(x)
            time.append(y) 

        return reg_anal_res, lab_vals, time

    def analysis_func(self, lab_measurements, top200_meds, n_medlab_pairs = 25, n_meds=None):
        res = pd.DataFrame(columns=['Medication','Lab Test','Number of patients','Estimated (mean)','Estimated (std)', 'Feature After(mean)','Feature After(std)','Time After(mean)','Time After(std)','Mannwhitney-pvalue','Ttest-pvalue'])
        uniqueLabTests = lab_measurements.LABEL.unique()
        final_res = []
        after_vals = []

        for i, med in enumerate(top200_meds['MED']): 
            if n_meds is not None and i>=n_meds:
                break
            print(i, ' MED: ', med)
            for j in tqdm(range(uniqueLabTests.shape[0])):
                labTest = uniqueLabTests[j]
                drug_lab, before, after = labpairing(med, patient_presc, lab_measurements, labTest)
                subjects = before['SUBJECT_ID'].unique()
                if(len(before) > n_medlab_pairs):
                    before_reg_anal_res, before_lab_vals, before_time = gen_estimate_coef(subjects, before)
                    after_reg_anal_res, after_lab_vals, after_time = gen_estimate_coef(subjects, after)
                    estimated = np.array(pd.DataFrame(after_reg_anal_res)['estimated'])
                    after_values = np.array([k.mean() for k in after_lab_vals])
                    ttest_res = stats.ttest_ind(estimated, after_values)[1]
                    mannwhitneyu_res = stats.mannwhitneyu(estimated, after_values)[1]

                    before_values1 = np.array(pd.DataFrame(before_reg_anal_res)['coef'])
                    after_values1 = np.array(pd.DataFrame(after_reg_anal_res)['coef'])
                    ttest_res1 = stats.ttest_ind(before_values1, after_values1)[1]
                    mannwhitneyu_res1 = stats.mannwhitneyu(before_values1, after_values1)[1]

                    final_res.append([med, labTest, len(before), np.mean(estimated), np.std(estimated), np.mean(after_values), np.std(after_values), np.mean(np.array([k.mean() for k in after_time])), np.std(np.array([k.mean() for k in after_time])), ttest_res, mannwhitneyu_res, np.mean(before_values1), np.mean(after_values1), ttest_res1, mannwhitneyu_res1])
        return pd.DataFrame(final_res, columns=['Medication','Lab Test', 'Number of patients', 'Estimated (mean)','Estimated (std)', 'Lab Test After(mean)','Lab Test After(std)','Time After(mean)','Time After(std)', 'Ttest-pvalue', 'Mannwhitney-pvalue', 'Before','After','Coef-Ttest-pvalue', 'Coef-Mannwhitney-pvalue']), after_vals

# # Age based stratification

# inputevents_mv1['STARTTIME'] = pd.to_datetime(inputevents_mv1['STARTTIME'],  format='%Y/%m/%d %H:%M:%S')
# inputevents_mv1['ENDTIME'] = pd.to_datetime(inputevents_mv1['ENDTIME'],  format='%Y/%m/%d %H:%M:%S')

# patients

# patients_with_dob = pd.merge(inputevents_mv1, patients, how='inner', on='SUBJECT_ID')

# patients_with_dob['DOB'] = pd.to_datetime(patients_with_dob['DOB'],  format='%Y/%m/%d %H:%M:%S')
# patients_with_dob['AGE'] = (patients_with_dob['STARTTIME'].dt.date - patients_with_dob['DOB'].dt.date).dt.years
# patients_with_age = patients_with_dob.drop(columns=['Unnamed: 0', 'DOD_HOSP', 'DOD_SSN', 'DOD', 'ROW_ID'])

# temp = patients_with_dob['STARTTIME'].dt.date - patients_with_dob['DOB'].dt.date

# temp.apply()

# temp.apply(lambda t : pd.to_timedelta(t, 'days').days/365 )



# # Other Analysis

# res_analysis = pd.read_csv(os.path.join(RESULT, 'Round-3', 'Prescription-Analysis-1.csv'))

# res_analysis

# presc = pd.read_csv(os.path.join(RESULT, 'prescription_preprocessed.csv'))
# presc['STARTDATE'] = pd.to_datetime(presc['STARTDATE'],  format='%Y/%m/%d %H:%M:%S')
# presc['ENDDATE'] = pd.to_datetime(presc['ENDDATE'],  format='%Y/%m/%d %H:%M:%S')

# def labpairing(medname, prescdf, labdf, labname, k=3):
#     '''
#     Pairs the drug input with each lab test

#     Parameters:
#     drugname (String): Drug Name
#     prescdf (DataFrame): Dataframe containing the prescription data
#     labdf (DataFrame): Dataframe containing the lab measurement data
#     labname (DataFrame): Lab Test Name
#     Returns:
#     DataFrame: Contains all the rows of values and times for that particular drug lab apir
#     '''
    
#     # Select patients who have taken the drug
#     prescdf = prescdf[prescdf['DRUG']==medname]
#     prescdf = prescdf.drop_duplicates(subset=['SUBJECT_ID'], keep='first')

#     # Select lab measurements of patients who have taken the drug
#     labdf = labdf[labdf['HADM_ID'].isin(prescdf['HADM_ID'])]

#     # Selects the lab measurement entered
#     drug_lab_specific = labdf[labdf['LABEL']==labname]
#     mergeddf = pd.merge(drug_lab_specific, prescdf, on=['HADM_ID','SUBJECT_ID'])

#     # Get time from prescription and choose before and after lab measurements (within 24hrs=1day)
#     mergeddf['timeFromPrescription'] = mergeddf['CHARTTIME'] - mergeddf['STARTDATE']
#     mergeddf = mergeddf[(mergeddf['timeFromPrescription']>datetime.timedelta(days=(-1*k))) & (mergeddf['timeFromPrescription']<datetime.timedelta(days=k))]
#     posmergeddf = mergeddf.loc[mergeddf.timeFromPrescription > datetime.timedelta(hours=12)]
#     negmergeddf = mergeddf.loc[mergeddf.timeFromPrescription < datetime.timedelta(hours=-12)]
    
#     # Only keep values for which we have both before and after
#     posmergeddf = posmergeddf[posmergeddf['HADM_ID'].isin(negmergeddf['HADM_ID'])]
#     negmergeddf = negmergeddf[negmergeddf['HADM_ID'].isin(posmergeddf['HADM_ID'])]
#     df = negmergeddf.merge(posmergeddf,on=['HADM_ID','SUBJECT_ID'])

#     # Choose admissions which have more than one lab test reading
#     before = negmergeddf
#     bool_before = before.groupby('SUBJECT_ID').count()>1
#     index_before = bool_before[bool_before['HADM_ID']==True].index
#     before = before[before['SUBJECT_ID'].isin(index_before)]
#     negmergeddf = negmergeddf.loc[negmergeddf.groupby('SUBJECT_ID').timeFromPrescription.idxmax()]

#     after = posmergeddf
#     bool_after = after.groupby('SUBJECT_ID').count()>1
#     index_after = bool_after[bool_after['HADM_ID']==True].index
#     after = after[after['SUBJECT_ID'].isin(index_after)]
#     posmergeddf = posmergeddf.loc[posmergeddf.groupby('SUBJECT_ID').timeFromPrescription.idxmin()]

#     before = before[before['HADM_ID'].isin(after['HADM_ID'])]
#     after = after[after['HADM_ID'].isin(before['HADM_ID'])]

#     finaldf = negmergeddf.merge(posmergeddf,on=['HADM_ID','SUBJECT_ID'])
    
#     return finaldf, before, after

# pairs, befores, afters = [], [], []
# for row in res_analysis.iterrows():
#     if row[0]<10:
#         pair, before, after = labpairing(row[1]['Medication'], presc, lab_measurements, row[1]['Lab Test'])
#         pairs.append(pair)
#         befores.append(before)
#         afters.append(after)
#     else:
#         break

# mean_vals = []
# time_vals = []
# for before in befores:
#     mean_vals.append(before.groupby('SUBJECT_ID').mean()['VALUENUM'].mean())
# for before in befores:
#     time_vals.append(before['timeFromPrescription'].apply(lambda x : x.total_seconds()/3600 ).mean())

# d = {
#     'Lab Test Before (mean)' : mean_vals,
#      'Time Before (mean)' : time_vals
# }
# pd.DataFrame(d)

# before.groupby('SUBJECT_ID').mean()['VALUENUM'].mean()

# before['timeFromPrescription'].apply(lambda x : x.total_seconds()/3600 )
# # before.groupby('SUBJECT_ID').mean()

# after.groupby('SUBJECT_ID').mean()['VALUENUM'].mean()

# presc[presc['SUBJECT_ID']==6]['HADM_ID'].unique()

# lab_measurements[lab_measurements['SUBJECT_ID']==6]['HADM_ID'].unique()

# def get_min(col):
#     return col.apply(lambda td : td.total_seconds()//60)

# def get_hour(col):
#     return col.apply(lambda td : td.total_seconds()//3600)

# # Regression Analysis
# def gen_estimate_coef(subjects, before, verbose=False):
#     reg_anal_res = []
#     lab_vals = []
#     time = []

#     for i in subjects:
        
#         res_vals = dict()
#         res_vals['subjectID'] = i
        
#         rows = before[before['SUBJECT_ID']==i]
#         rows = rows.sort_values(by='timeFromPrescription')

#         x = rows['VALUENUM']
#         y = get_min(rows['timeFromPrescription'])

#         reg = linear_model.LinearRegression()
#         reg.fit(np.array(y).reshape(-1,1), np.array(x).reshape(-1,1))

#         res_vals['coef'] = reg.coef_[0][0]
#         res_vals['estimated'] = reg.predict([[0]])[0][0]
        
#         if verbose:
#             print('Subject ID: ', i, '  Coefficients: ', reg.coef_)

#         reg_anal_res.append(res_vals)

#         lab_vals.append(x)
#         time.append(y) 

#     return reg_anal_res, lab_vals, time

# def analysis_func(labTest, med, n_medlab_pairs = 25, n_meds=None):
#     final_res = []
#     after_vals = []

#     drug_lab, before, after = labpairing(med, presc, lab_measurements, labTest)
#     subjects = before['SUBJECT_ID'].unique()
#     if(len(before) > n_medlab_pairs):
#         before_reg_anal_res, before_lab_vals, before_time = gen_estimate_coef(subjects, before)
#         after_reg_anal_res, after_lab_vals, after_time = gen_estimate_coef(subjects, after)
#         estimated = np.array(pd.DataFrame(before_reg_anal_res)['estimated'])
        
#         before_values = np.array([k.mean() for k in before_lab_vals])
#         after_values = np.array([k.mean() for k in after_lab_vals])

#         # Befoer and after absolute values
#         ttest_res0 = stats.ttest_ind(estimated, before_values)[1]
#         mannwhitneyu_res0 = stats.mannwhitneyu(estimated, after_values)[1]

#         # Estimated value after regression and after medication absolute values
#         ttest_res = stats.ttest_ind(estimated, after_values)[1]
#         mannwhitneyu_res = stats.mannwhitneyu(estimated, after_values)[1]

#         # Befoer and after regression coefficient values
#         before_values1 = np.array(pd.DataFrame(before_reg_anal_res)['coef'])
#         after_values1 = np.array(pd.DataFrame(after_reg_anal_res)['coef'])
#         ttest_res1 = stats.ttest_ind(before_values1, after_values1)[1]
#         mannwhitneyu_res1 = stats.mannwhitneyu(before_values1, after_values1)[1]

#         return [med, labTest, len(before), np.mean(estimated), np.std(estimated), np.mean(after_values), np.std(after_values), np.mean(np.array([k.mean() for k in after_time])), np.std(np.array([k.mean() for k in after_time])), ttest_res0, mannwhitneyu_res0, ttest_res, mannwhitneyu_res, np.mean(before_values1), np.mean(after_values1), ttest_res1, mannwhitneyu_res1]

# res = []
# for row in res_analysis.iterrows():
#     if row[0]<10:
#         res.append(analysis_func(row[1]['Lab Test'], row[1]['Medication'], n_medlab_pairs = 200))
#     else:
#         break

# import random
# random.seed(10)
# subjects_2k = random.sample(list(temp['SUBJECT_ID'].value_counts().keys()), 15000)

# first_admissions = admissions.drop_duplicates(subset=['SUBJECT_ID'], keep='first')
# first_admissions[first_admissions]

# subs = list(set(presc['HADM_ID'].unique()).intersection(set(first_admissions['HADM_ID'].unique())))
# presc['SUBJECT_ID'] = presc['SUBJECT_ID'].astype('int')
# p = presc[presc['SUBJECT_ID'].isin(subs)]
# p

# k = presc['SUBJECT_ID'].isin(subs)

# presc['HADM_ID'].unique().shape

# presc[presc['DRUG']=='Warfarin'].tail(30)

# pd.DataFrame(res, columns=['Medication','Lab Test', 'Number of patients', 'Estimated (mean)','Estimated (std)', 'Lab Test After(mean)','Lab Test After(std)','Time After(mean)','Time After(std)', 'Absolute-Ttest-pvalue', 'Absolute-Mannwhitney-pvalue', 'Ttest-pvalue', 'Mannwhitney-pvalue', 'Before','After', 'Coef-Ttest-pvalue', 'Coef-Mannwhitney-pvalue'])

# # Lab VS Time difference Plot

# ## Inputevents


# patient_presc = inputevents_mv1
# lab_measurements = labValues



# finalDF, before1, after1 = labpairing('Insulin - Regular', patient_presc, lab_measurements, 'Glucose')

# before1

# drug_lab = finalDF

# top200_meds = inputevents_mv1['LABEL'].value_counts()[:200]

# pd.DataFrame(top200_meds).head(20)

# def get_min(col):
#     return col.apply(lambda td : td.total_seconds()//60)
# def get_hour(col):
#     return col.apply(lambda td : td.total_seconds()//3600)

# reg_anal_res = []
# lab_vals = []
# time = []
# before = before1
# subjects = list(drug_lab['SUBJECT_ID'].unique())

# for i in subjects:
    
#     res_vals = dict()
#     res_vals['subjectID'] = i
    
#     rows = before[before['SUBJECT_ID']==i]
#     rows = rows.sort_values(by='timeFromPrescription')

#     x = rows['VALUENUM']
#     y = get_hour(rows['timeFromPrescription'])

#     if np.array(x).shape[0]>0 and np.array(y).shape[0]>0:
#         reg = linear_model.LinearRegression()
#         reg.fit(np.array(y).reshape(-1,1), np.array(x).reshape(-1,1))

#         res_vals['coef'] = reg.coef_[0][0]
#         res_vals['estimated'] = reg.predict([[0]])[0][0]

#         reg_anal_res.append(res_vals)

#         lab_vals.append(x)
#         time.append(y)

# e = pd.DataFrame(reg_anal_res)
# e = e.rename(columns={'subjectID':'SUBJECT_ID'})

# after1['timeFromPrescription'] = after1['timeFromPrescription'].apply(lambda x : round(x.total_seconds()/3600, 2) )

# # merged = pd.merge(drug_lab, e, how='inner', on='SUBJECT_ID')
# absolute = after1.groupby('SUBJECT_ID').mean().reset_index()['VALUENUM']-e['estimated']

# time_diff = after1.groupby('SUBJECT_ID').mean().reset_index()['timeFromPrescription']

# time_diff

# # time_diff = time_diff.apply(lambda t : t.total_seconds()/3600)

# after = after1.groupby('SUBJECT_ID').mean().reset_index()['VALUENUM']
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
#     plot_data = plot_data[plot_data['timeFromPrescription']>1 & plot_data['timeFromPrescription']<24]
#     sns.regplot(x = "timeFromPrescription", 
#             y = 'Lab values', 
#             data = plot_data, 
#             truncate=False)
#     plt.title('Insulin<>Glucose - '+ title+ ' change in lab measurment and time taken for change')
#     plt.xlabel('Time in hours')
#     plt.ylabel('Glucose Levels (mg/dL)')
#     plt.show()

# absolute1.reset_index()

# absolute1, time_diff3 = remove_outlier(absolute, time_diff)
# plot_func(absolute1, time_diff3, 'Absolute')

# percent1, time_diff1 = remove_outlier(percent, time_diff)
# plot_func(percent1, time_diff1, 'Percentage')

# ratio1, time_diff2 = remove_outlier(ratio, time_diff)
# plot_func(ratio1, time_diff2, 'Ratio')



# ## Prescriptions

# presc = pd.read_csv(os.path.join(RESULT, 'prescription_preprocessed.csv'))
# presc['STARTDATE'] = pd.to_datetime(presc['STARTDATE'],  format='%Y/%m/%d %H:%M:%S')
# presc['ENDDATE'] = pd.to_datetime(presc['ENDDATE'],  format='%Y/%m/%d %H:%M:%S')

# def labpairing(medname, prescdf, labdf, labname, k=3):
#     '''
#     Pairs the drug input with each lab test

#     Parameters:
#     drugname (String): Drug Name
#     prescdf (DataFrame): Dataframe containing the prescription data
#     labdf (DataFrame): Dataframe containing the lab measurement data
#     labname (DataFrame): Lab Test Name
#     Returns:
#     DataFrame: Contains all the rows of values and times for that particular drug lab apir
#     '''
    
#     # Select patients who have taken the drug
#     prescdf = prescdf[prescdf['DRUG']==medname]
#     prescdf = prescdf.drop_duplicates(subset=['SUBJECT_ID'], keep='first')

#     # Select lab measurements of patients who have taken the drug
#     labdf = labdf[labdf['HADM_ID'].isin(prescdf['HADM_ID'])]

#     # Selects the lab measurement entered
#     drug_lab_specific = labdf[labdf['LABEL']==labname]
#     mergeddf = pd.merge(drug_lab_specific, prescdf, on=['HADM_ID','SUBJECT_ID'])

#     # Get time from prescription and choose before and after lab measurements (within 24hrs=1day)
#     mergeddf['timeFromPrescription'] = mergeddf['CHARTTIME'] - mergeddf['STARTDATE']
#     mergeddf = mergeddf[(mergeddf['timeFromPrescription']>datetime.timedelta(days=(-1*k))) & (mergeddf['timeFromPrescription']<datetime.timedelta(days=k))]
#     posmergeddf = mergeddf.loc[mergeddf.timeFromPrescription > datetime.timedelta(hours=12)]
#     negmergeddf = mergeddf.loc[mergeddf.timeFromPrescription < datetime.timedelta(hours=-12)]
    
#     # Only keep values for which we have both before and after
#     posmergeddf = posmergeddf[posmergeddf['HADM_ID'].isin(negmergeddf['HADM_ID'])]
#     negmergeddf = negmergeddf[negmergeddf['HADM_ID'].isin(posmergeddf['HADM_ID'])]
#     df = negmergeddf.merge(posmergeddf,on=['HADM_ID','SUBJECT_ID'])

#     # Choose admissions which have more than one lab test reading
#     before = negmergeddf
#     bool_before = before.groupby('SUBJECT_ID').count()>1
#     index_before = bool_before[bool_before['HADM_ID']==True].index
#     before = before[before['SUBJECT_ID'].isin(index_before)]
#     negmergeddf = negmergeddf.loc[negmergeddf.groupby('SUBJECT_ID').timeFromPrescription.idxmax()]

#     after = posmergeddf
#     bool_after = after.groupby('SUBJECT_ID').count()>1
#     index_after = bool_after[bool_after['HADM_ID']==True].index
#     after = after[after['SUBJECT_ID'].isin(index_after)]
#     posmergeddf = posmergeddf.loc[posmergeddf.groupby('SUBJECT_ID').timeFromPrescription.idxmin()]

#     before = before[before['HADM_ID'].isin(after['HADM_ID'])]
#     after = after[after['HADM_ID'].isin(before['HADM_ID'])]

#     finaldf = negmergeddf.merge(posmergeddf,on=['HADM_ID','SUBJECT_ID'])
    
#     return finaldf, before, after

# patient_presc = presc
# lab_measurements = labValues
# finalDF, before1, after1 = labpairing('Insulin', patient_presc, lab_measurements, 'Glucose')

# drug_lab = finalDF

# top200_meds = presc['DRUG'].value_counts()[:100]

# pd.DataFrame(top200_meds).head(20)

# drug_lab

# def get_min(col):
#     return col.apply(lambda td : td.total_seconds()//60)
# def get_hour(col):
#     return col.apply(lambda td : td.total_seconds()//3600)

# reg_anal_res = []
# lab_vals = []
# time = []
# before = before1
# subjects = list(drug_lab['SUBJECT_ID'])

# for i in subjects:
    
#     res_vals = dict()
#     res_vals['subjectID'] = i
    
#     rows = before[before['SUBJECT_ID']==i]
#     rows = rows.sort_values(by='timeFromPrescription')

#     x = rows['VALUENUM']
#     y = get_hour(rows['timeFromPrescription'])

#     if np.array(x).shape[0]>0 and np.array(y).shape[0]>0:
#         reg = linear_model.LinearRegression()
#         reg.fit(np.array(y).reshape(-1,1), np.array(x).reshape(-1,1))

#         res_vals['coef'] = reg.coef_[0][0]
#         res_vals['estimated'] = reg.predict([[0]])[0][0]

#         reg_anal_res.append(res_vals)

#         lab_vals.append(x)
#         time.append(y)

# e = pd.DataFrame(reg_anal_res)
# e = e.rename(columns={'subjectID':'SUBJECT_ID'})

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

# absolute1, time_diff3 = remove_outlier(absolute, time_diff)
# plot_func(absolute1, time_diff3, 'Absolute')

# percent1, time_diff1 = remove_outlier(percent, time_diff)
# plot_func(percent1, time_diff1, 'Percentage')

# ratio1, time_diff2 = remove_outlier(ratio, time_diff)
# plot_func(ratio1, time_diff2, 'Ratio')

if __name__=="__main__":
    print('Start')
    CURR_DIR = os.getcwd()
    BASE_DIR = os.path.dirname(CURR_DIR)
    print(CURR_DIR, BASE_DIR)
    data = Dataset('mimiciii', os.path.join(BASE_DIR, 'data'))
    