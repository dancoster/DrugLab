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

# # Age based stratification

# inputevents_mv1['STARTTIME'] = pd.to_datetime(inputevents_mv1['STARTTIME'],  format='%Y/%m/%d %H:%M:%S')
# inputevents_mv1['ENDTIME'] = pd.to_datetime(inputevents_mv1['ENDTIME'],  format='%Y/%m/%d %H:%M:%S')

# patients

# patients_with_dob = pd.merge(inputevents_mv1, patients, how='inner', on='SUBJECT_ID')

# patients_with_dob['DOB'] = pd.to_datetime(patients_with_dob['DOB'],  format='%Y/%m/%d %H:%M:%S')
# patients_with_dob['AGE'] = (patients_with_dob['STARTTIME'].dt.date - patients_with_dob['DOB'].dt.date).dt.years
# patients_with_age = patients_with_dob.drop(columns=['Unnamed: 0', 'DOD_HOSP', 'DOD_SSN', 'DOD', 'ROW_ID'])

# temp = patients_with_dob['STARTTIME'].dt.date - patients_with_dob['DOB'].dt.date


if __name__=="__main__":
    print('Start')
    CURR_DIR = os.getcwd()
    BASE_DIR = os.path.dirname(CURR_DIR)
    print(CURR_DIR, BASE_DIR)
    data = Dataset('mimiciii', os.path.join(BASE_DIR, 'data'))
    