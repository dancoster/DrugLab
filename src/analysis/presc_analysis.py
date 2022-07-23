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


class PrescriptionsDataAnalysis:

    def __init__(self, path, dataset):
        self.RESULTS = path
        self.data = dataset

    def analyse(self, data_type , n=200):

        ## Generating Lab Test<>Meds Pairings
        finalDF, before, after = PrescriptionsDataAnalysis.labpairing('NaCl 0.9%', patient_presc, lab_measurements, 'Calcium, Total')

        ## Final Results - Reading before and after, regression and trend
        res = self.before_after_generator(self.lab_measurements, self.meds, n)
        res.to_csv(os.path.join(self.RESULTS, 'prescriptions_before_after_regression_trend.csv'))

    @staticmethod
    def labpairing(self, medname, prescdf, labdf, labname, k=3):
        '''
        Pairs the drug input with each lab test

        Parameters:
        drugname (String): Drug Name
        prescdf (DataFrame): Dataframe containing the prescription data
        labdf (DataFrame): Dataframe containing the lab measurement data
        labname (DataFrame): Lab Test Name
        Returns:
        DataFrame: Contains all the rows of values and times for that particular drug lab apir
        '''
        
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
    
    def results_generator(self, lab_measurements, top200_meds, n_medlab_pairs = 25, n_meds=None):
        uniqueLabTests = lab_measurements.LABEL.unique()
        final_res = []
        after_vals = []

        for i, med in enumerate(top200_meds['DRUG']): 
            if n_meds is not None and i>=n_meds:
                break
            print(i, ' MED: ', med)
            for j in tqdm(range(uniqueLabTests.shape[0])):
                labTest = uniqueLabTests[j]
                drug_lab, _, after, before = labpairing(med, patient_presc, lab_measurements, labTest)
                subjects = before['SUBJECT_ID'].unique()
                if(len(before) > n_medlab_pairs):
                    before_reg_anal_res, before_lab_vals, before_time = self.gen_estimate_coef(subjects, before)
                    after_reg_anal_res, after_lab_vals, after_time = self.gen_estimate_coef(subjects, after)
                    estimated = np.array(pd.DataFrame(before_reg_anal_res)['estimated'])
                    
                    before_values = np.array([k.mean() for k in before_lab_vals])
                    after_values = np.array([k.mean() for k in after_lab_vals])

                    # Befoer and after absolute values
                    ttest_res0 = stats.ttest_ind(before_values, after_values)[1]
                    mannwhitneyu_res0 = stats.mannwhitneyu(before_values, after_values)[1]

                    # Estimated value after regression and after medication absolute values
                    ttest_res = stats.ttest_ind(estimated, after_values)[1]
                    mannwhitneyu_res = stats.mannwhitneyu(estimated, after_values)[1]

                    # Befoer and after regression coefficient values
                    before_values1 = np.array(pd.DataFrame(before_reg_anal_res)['coef'])
                    after_values1 = np.array(pd.DataFrame(after_reg_anal_res)['coef'])
                    ttest_res1 = stats.ttest_ind(before_values1, after_values1)[1]
                    mannwhitneyu_res1 = stats.mannwhitneyu(before_values1, after_values1)[1]

                    final_res.append([med, labTest, len(before), np.mean(estimated), np.std(estimated), np.mean(after_values), np.std(after_values), np.mean(np.array([k.mean() for k in after_time])), np.std(np.array([k.mean() for k in after_time])), ttest_res0, mannwhitneyu_res0, ttest_res, mannwhitneyu_res, np.mean(before_values1), np.mean(after_values1), ttest_res1, mannwhitneyu_res1])
        return pd.DataFrame(final_res, columns=['Medication','Lab Test', 'Number of patients', 'Estimated (mean)','Estimated (std)', 'Lab Test After(mean)','Lab Test After(std)','Time After(mean)','Time After(std)', 'Absolute-Ttest-pvalue', 'Absolute-Mannwhitney-pvalue', 'Ttest-pvalue', 'Mannwhitney-pvalue', 'Before','After', 'Coef-Ttest-pvalue', 'Coef-Mannwhitney-pvalue']), after_vals

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
