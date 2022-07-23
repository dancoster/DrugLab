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

class PRDataAnalysis(Analysis):

    def __init__(self, path, dataset):
        self.logger = logging.getLogger(self.__class__.__name__)
        Analysis.__init__(self, path, dataset, 'prescriptions')

    def results_generator(self, med, patient_presc, lab_measurements, labTest, n_medlab_pairs=2000):
        drug_lab, before, after = Analysis.labpairing(med, patient_presc, lab_measurements, labTest, type=self.type)
        subjects = before['SUBJECT_ID'].unique()
        if(len(subjects) > n_medlab_pairs):
            before_reg_anal_res, before_lab_vals, before_time = Analysis.interpolation(subjects, before)
            after_reg_anal_res, after_lab_vals, after_time = Analysis.interpolation(subjects, after)
            estimated = np.array(pd.DataFrame(before_reg_anal_res)['estimated'])
            
            before_values = np.array([list(k)[0] for k in before_lab_vals])
            after_values = np.array([list(k)[0] for k in after_lab_vals])

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

            return [med, labTest, len(before), np.mean(estimated), np.std(estimated), np.mean(after_values), np.std(after_values), np.mean(np.array([k.mean() for k in after_time])), np.std(np.array([k.mean() for k in after_time])), ttest_res0, mannwhitneyu_res0, ttest_res, mannwhitneyu_res, np.mean(before_values1), np.mean(after_values1), ttest_res1, mannwhitneyu_res1]
        
        return None

    def results_analysis(self, patient_presc, lab_measurements, meds, n_medlab_pairs = 25, n_meds=None):
        uniqueLabTests = lab_measurements.LABEL.unique()
        final_res = []
        after_vals = []

        for i, med in enumerate(meds['MED']): 
            if n_meds is not None and i>=n_meds:
                break
            print(i, ' MED: ', med)
            for j in tqdm(range(uniqueLabTests.shape[0])):
                labTest = uniqueLabTests[j]
                row = self.results_generator(med, patient_presc, lab_measurements, labTest, n_medlab_pairs)
                if row is not None:
                    final_res.append(row)
        return pd.DataFrame(final_res, columns=['Medication','Lab Test', 'Number of patients', 'Estimated (mean)','Estimated (std)', 'Lab Test After(mean)','Lab Test After(std)','Time After(mean)','Time After(std)', 'Absolute-Ttest-pvalue', 'Absolute-Mannwhitney-pvalue', 'Ttest-pvalue', 'Mannwhitney-pvalue', 'Before','After', 'Coef-Ttest-pvalue', 'Coef-Mannwhitney-pvalue'])