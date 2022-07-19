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


class InputeventsDataAnalysis:

    def __init__(self, path):
        self.RESULTS = path

    def run(self, data_type):

        ## Generating Lab Test<>Meds Pairings
        finalDF, before, after = labpairing('NaCl 0.9%', patient_presc, lab_measurements, 'Calcium, Total')

        ## Final Results - Reading before and after
        res = self.before_after_generator(lab_measurements, top200_meds[:60], 200)
        res.to_csv(os.path.join(self.RESULTS, 'inputevents_before_after.csv'))
            
        ## Final Results - Regression and trend
        final_res_df = self.reg_trend_generator(lab_measurements, top200_meds[:60], n_medlab_pairs = 200)
        final_res_df.to_csv(os.path.join(self.RESULTS, 'inputevents_regression_trend.csv'))

    def labpairing(self, medname, prescdf, labdf, labname, k=3):
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

     def before_after_generator(self, lab_measurements, top100_drugs, n_druglab_pairs = 25, n_drugs=None):
        '''
        Final Results - Before and After
        '''
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
    def get_min(self, col):
        return col.apply(lambda td : td.total_seconds()//60)

    def get_hour(self, col):
        return col.apply(lambda td : td.total_seconds()//3600)

    # Regression Analysis
    def gen_estimate_coef(self, subjects, before, verbose=False):
        reg_anal_res = []
        lab_vals = []
        time = []

        for i in subjects:
            
            res_vals = dict()
            res_vals['subjectID'] = i
            
            rows = before[before['SUBJECT_ID']==i]
            rows = rows.sort_values(by='timeFromPrescription')

            x = rows['VALUENUM']
            y = self.get_min(rows['timeFromPrescription'])

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

    def reg_trend_generator(self, labTest, med, n_medlab_pairs = 25, n_meds=None):
        '''
        Final results generator - Regression and trend analysis
        '''
        final_res = []
        after_vals = []

        drug_lab, before, after = labpairing(med, presc, lab_measurements, labTest)
        subjects = before['SUBJECT_ID'].unique()
        if(len(before) > n_medlab_pairs):
            before_reg_anal_res, before_lab_vals, before_time = self.gen_estimate_coef(subjects, before)
            after_reg_anal_res, after_lab_vals, after_time = self.gen_estimate_coef(subjects, after)
            estimated = np.array(pd.DataFrame(before_reg_anal_res)['estimated'])
            
            before_values = np.array([k.mean() for k in before_lab_vals])
            after_values = np.array([k.mean() for k in after_lab_vals])

            # Befoer and after absolute values
            ttest_res0 = stats.ttest_ind(estimated, before_values)[1]
            mannwhitneyu_res0 = stats.mannwhitneyu(estimated, after_values)[1]

            # Estimated value after regression and after medication absolute values
            ttest_res = stats.ttest_ind(estimated, after_values)[1]
            mannwhitneyu_res = stats.mannwhitneyu(estimated, after_values)[1]

            # Befoer and after regression coefficient values
            before_values1 = np.array(pd.DataFrame(before_reg_anal_res)['coef'])
            after_values1 = np.array(pd.DataFrame(after_reg_anal_res)['coef'])
            ttest_res1 = stats.ttest_ind(before_values1, after_values1)[1]
            mannwhitneyu_res1 = stats.mannwhitneyu(before_values1, after_values1)[1]

            return [med, labTest, len(before), np.mean(estimated), np.std(estimated), np.mean(after_values), np.std(after_values), np.mean(np.array([k.mean() for k in after_time])), np.std(np.array([k.mean() for k in after_time])), ttest_res0, mannwhitneyu_res0, ttest_res, mannwhitneyu_res, np.mean(before_values1), np.mean(after_values1), ttest_res1, mannwhitneyu_res1]

