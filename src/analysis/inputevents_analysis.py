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
from analysis.analysis import Analysis

logging.basicConfig(level=logging.INFO, format=f'%(module)s/%(filename)s [Class: %(name)s Func: %(funcName)s] %(levelname)s : %(message)s')

class IEDataAnalysis(Analysis):

    def __init__(self, path, dataset, suffix=''):
        self.logger = logging.getLogger(self.__class__.__name__)
        Analysis.__init__(self, path, dataset, 'inputevents', suffix=suffix)

    def analyse_custom(self, n_subs=200, n_meds=50):

        patient_presc = self.data.patient_presc[self.table]
        lab_measurements = self.data.lab_measurements[self.table]
        meds = self.data.meds[self.table]

        ## Generating Lab Test<>Meds Pairings
        self.logger.info(f'Testing the working on labpairing generation for {self.table} data...')
        finalDF, before, after = Analysis.labpairing('NaCl 0.9%', patient_presc, lab_measurements, 'Calcium, Total')
        self.logger.info(f'Testing successful.')

        ## Final Results - Reading one lab test before and after analysis
        self.logger.info(f'Performing analysis of medication effect, by comparing two labtest values (one taken before and another taken after) using statistical hypothesis testing for {self.table} data...')
        res = self.before_after_analysis(patient_presc, lab_measurements, meds, n_subs, n_meds)
        time = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
        res.to_csv(os.path.join(self.RESULTS, 'inputevents_before_after_'+time+'.csv'))
        self.logger.info(f'Analysis done for {self.table} data. Stored data in {self.RESULTS}')
            
        ## Final Results - Interpolation analysis and trend analysis
        self.logger.info(f'Performing analysis of medication effect, by "comparing the interpolated labtest value at time of medication and after labtest value" and "before and after trends" using statistical hypothesis testing for {self.table} data...')
        final_res_df = self.reg_trend_analysis(patient_presc, lab_measurements, meds, n_subs, n_meds)
        time = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
        final_res_df.to_csv(os.path.join(self.RESULTS, 'inputevents_regression_trend_'+time+'.csv'))
        self.logger.info(f'Analysis done for {self.table} data. Stored data in {self.RESULTS}')

    def before_after_generator(self, drug, patient_presc, lab_measurements, labTest,  n_druglab_pairs=25):
        '''
        Before and After statistical test values.
        '''        
        drug_lab, before, after = Analysis.labpairing(drug, patient_presc, lab_measurements, labTest)
        
        num = drug_lab['SUBJECT_ID'].unique().shape[0]
        before_num = before['SUBJECT_ID'].unique().shape[0]
        after_num = after['SUBJECT_ID'].unique().shape[0]
        
        if num > n_druglab_pairs and before_num > (0.25*n_druglab_pairs) and after_num > (0.25*n_druglab_pairs): 
            
            drug_lab['timeFromPrescription_x'] = pd.to_numeric(drug_lab['timeFromPrescription_x'].dt.seconds)
            drug_lab['timeFromPrescription_x']/=3600
            drug_lab['timeFromPrescription_y'] = pd.to_numeric(drug_lab['timeFromPrescription_y'].dt.seconds)
            drug_lab['timeFromPrescription_y']/=3600
            df_before_mean = drug_lab['VALUENUM_x'].mean()
            df_after_mean = drug_lab['VALUENUM_y'].mean()
            df_before_std = drug_lab['VALUENUM_x'].std()
            df_after_std = drug_lab['VALUENUM_y'].std()
            df_before_time_mean = drug_lab['timeFromPrescription_x'].mean()
            df_after_time_mean = drug_lab['timeFromPrescription_y'].mean()
            df_before_time_std = drug_lab['timeFromPrescription_x'].std()
            df_after_time_std = drug_lab['timeFromPrescription_y'].std()

            ttestpvalue = stats.ttest_ind(drug_lab['VALUENUM_x'], drug_lab['VALUENUM_y'])[1]
            mannwhitneyu = stats.mannwhitneyu(drug_lab['VALUENUM_x'], drug_lab['VALUENUM_y'])[1]
            
            lengthofdf = num
            csvrow = [lengthofdf,df_before_mean,df_before_std,df_before_time_mean,df_before_time_std,df_after_mean,df_after_std,df_after_time_mean,df_after_time_std,ttestpvalue, mannwhitneyu]
            return csvrow
        return None

    def before_after_analysis(self, patient_presc, lab_measurements, meds, n_druglab_pairs = 25, n_drugs=None):
        '''
        Final Results - Before and After
        '''
        res = pd.DataFrame(columns=['Medication','Lab Test','Number of patients','Lab Test Before(mean)','Lab Test Before(std)','Time Before(mean)','Time Before(std)','Lab Test After(mean)','Lab Test After(std)','Time After(mean)','Time After(std)', 'Ttest-pvalue', 'Mannwhitney-pvalue'])
        uniqueLabTests = lab_measurements.LABEL.unique()

        for i, drug in enumerate(meds['MED']): 
            temp_med = meds[meds['MED']==drug]
            if temp_med['COUNT'].iloc[0]<n_drugs:
                break
            print(i, ' Medication: ', drug)
            for j in tqdm(range(uniqueLabTests.shape[0])):
                labTest = uniqueLabTests[j]
                csvrow = self.before_after_generator(drug, patient_presc, lab_measurements, labTest,  n_druglab_pairs)
                if csvrow is not None:
                    csvrow.insert(0, drug) 
                    csvrow.insert(1, labTest)
                    res.loc[len(res)] = csvrow
    
        return res

    def reg_trend_generator(self, med, presc, lab_measurements, labTest, n_medlab_pairs=200):
        '''
        Final results generator - Regression and trend analysis
        '''
        final_res = []
        after_vals = []

        drug_lab, before, after = Analysis.labpairing(med, presc, lab_measurements, labTest)
        subjects = before['SUBJECT_ID'].unique()
        
        num = drug_lab['SUBJECT_ID'].unique().shape[0]
        before_num = before['SUBJECT_ID'].unique().shape[0]
        after_num = after['SUBJECT_ID'].unique().shape[0]
        
        if num > n_medlab_pairs and before_num > (0.25*n_medlab_pairs) and after_num > (0.25*n_medlab_pairs): 
            
            before_reg_anal_res, before_lab_vals, before_time = Analysis.interpolation(subjects, before)
            after_reg_anal_res, after_lab_vals, after_time = Analysis.interpolation(subjects, after)
            estimated = np.array(pd.DataFrame(before_reg_anal_res)['estimated'])
            
            before_values = np.array([list(k)[-1] for k in before_lab_vals])
            after_values = np.array([list(k)[0] for k in after_lab_vals])

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

            return [med, labTest, num, np.mean(estimated), np.std(estimated), np.mean(after_values), np.std(after_values), np.mean(np.array([k.mean() for k in after_time])), np.std(np.array([k.mean() for k in after_time])), ttest_res0, mannwhitneyu_res0, ttest_res, mannwhitneyu_res, np.mean(before_values1), np.mean(after_values1), ttest_res1, mannwhitneyu_res1]
        
        return None