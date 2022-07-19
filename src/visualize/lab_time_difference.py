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



# Lab VS Time difference Plot

## Inputevents


patient_presc = inputevents_mv1
lab_measurements = labValues



finalDF, before1, after1 = labpairing('Insulin - Regular', patient_presc, lab_measurements, 'Glucose')

before1

drug_lab = finalDF

top200_meds = inputevents_mv1['LABEL'].value_counts()[:200]

pd.DataFrame(top200_meds).head(20)

def get_min(col):
    return col.apply(lambda td : td.total_seconds()//60)
def get_hour(col):
    return col.apply(lambda td : td.total_seconds()//3600)

reg_anal_res = []
lab_vals = []
time = []
before = before1
subjects = list(drug_lab['SUBJECT_ID'].unique())

for i in subjects:
    
    res_vals = dict()
    res_vals['subjectID'] = i
    
    rows = before[before['SUBJECT_ID']==i]
    rows = rows.sort_values(by='timeFromPrescription')

    x = rows['VALUENUM']
    y = get_hour(rows['timeFromPrescription'])

    if np.array(x).shape[0]>0 and np.array(y).shape[0]>0:
        reg = linear_model.LinearRegression()
        reg.fit(np.array(y).reshape(-1,1), np.array(x).reshape(-1,1))

        res_vals['coef'] = reg.coef_[0][0]
        res_vals['estimated'] = reg.predict([[0]])[0][0]

        reg_anal_res.append(res_vals)

        lab_vals.append(x)
        time.append(y)

e = pd.DataFrame(reg_anal_res)
e = e.rename(columns={'subjectID':'SUBJECT_ID'})

after1['timeFromPrescription'] = after1['timeFromPrescription'].apply(lambda x : round(x.total_seconds()/3600, 2) )

# merged = pd.merge(drug_lab, e, how='inner', on='SUBJECT_ID')
absolute = after1.groupby('SUBJECT_ID').mean().reset_index()['VALUENUM']-e['estimated']

time_diff = after1.groupby('SUBJECT_ID').mean().reset_index()['timeFromPrescription']

time_diff

# time_diff = time_diff.apply(lambda t : t.total_seconds()/3600)

after = after1.groupby('SUBJECT_ID').mean().reset_index()['VALUENUM']
estimate = e['estimated']

percent = 100*(after-estimate)/estimate

ratio = after/estimate

def remove_outlier(val, time_diff):
    val = pd.DataFrame(val)
    time_diff = pd.DataFrame(time_diff)
    # IQR
    Q1 = np.percentile(val, 25,
                    interpolation = 'midpoint')
    
    Q3 = np.percentile(val, 75,
                    interpolation = 'midpoint')
    IQR = Q3 - Q1
    
    # Upper bound
    upper = np.where(val >= (Q3+1.5*IQR))
    # Lower bound
    lower = np.where(val <= (Q1-1.5*IQR))
    val.drop(upper[0], inplace = True)
    time_diff.drop(upper[0], inplace = True)
    val.drop(lower[0], inplace = True)
    time_diff.drop(lower[0], inplace = True)
    return val, time_diff

import seaborn as sns

def plot_func(absolute, time_diff, title=''):
    plot_data = pd.concat([absolute, time_diff], axis=1)
    plot_data = plot_data.rename(columns={0:'Lab values'})
    plot_data = plot_data[plot_data['timeFromPrescription']>1 & plot_data['timeFromPrescription']<24]
    sns.regplot(x = "timeFromPrescription", 
            y = 'Lab values', 
            data = plot_data, 
            truncate=False)
    plt.title('Insulin<>Glucose - '+ title+ ' change in lab measurment and time taken for change')
    plt.xlabel('Time in hours')
    plt.ylabel('Glucose Levels (mg/dL)')
    plt.show()

absolute1.reset_index()

absolute1, time_diff3 = remove_outlier(absolute, time_diff)
plot_func(absolute1, time_diff3, 'Absolute')

percent1, time_diff1 = remove_outlier(percent, time_diff)
plot_func(percent1, time_diff1, 'Percentage')

ratio1, time_diff2 = remove_outlier(ratio, time_diff)
plot_func(ratio1, time_diff2, 'Ratio')



## Prescriptions

presc = pd.read_csv(os.path.join(RESULT, 'prescription_preprocessed.csv'))
presc['STARTDATE'] = pd.to_datetime(presc['STARTDATE'],  format='%Y/%m/%d %H:%M:%S')
presc['ENDDATE'] = pd.to_datetime(presc['ENDDATE'],  format='%Y/%m/%d %H:%M:%S')

def labpairing(medname, prescdf, labdf, labname, k=3):
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

patient_presc = presc
lab_measurements = labValues
finalDF, before1, after1 = labpairing('Insulin', patient_presc, lab_measurements, 'Glucose')

drug_lab = finalDF

top200_meds = presc['DRUG'].value_counts()[:100]

pd.DataFrame(top200_meds).head(20)

drug_lab

def get_min(col):
    return col.apply(lambda td : td.total_seconds()//60)
def get_hour(col):
    return col.apply(lambda td : td.total_seconds()//3600)

reg_anal_res = []
lab_vals = []
time = []
before = before1
subjects = list(drug_lab['SUBJECT_ID'])

for i in subjects:
    
    res_vals = dict()
    res_vals['subjectID'] = i
    
    rows = before[before['SUBJECT_ID']==i]
    rows = rows.sort_values(by='timeFromPrescription')

    x = rows['VALUENUM']
    y = get_hour(rows['timeFromPrescription'])

    if np.array(x).shape[0]>0 and np.array(y).shape[0]>0:
        reg = linear_model.LinearRegression()
        reg.fit(np.array(y).reshape(-1,1), np.array(x).reshape(-1,1))

        res_vals['coef'] = reg.coef_[0][0]
        res_vals['estimated'] = reg.predict([[0]])[0][0]

        reg_anal_res.append(res_vals)

        lab_vals.append(x)
        time.append(y)

e = pd.DataFrame(reg_anal_res)
e = e.rename(columns={'subjectID':'SUBJECT_ID'})

merged = pd.merge(drug_lab, e, how='inner', on='SUBJECT_ID')
absolute = merged['VALUENUM_y']-e['estimated']

time_diff = merged['timeFromPrescription_y']

time_diff = time_diff.apply(lambda t : t.total_seconds()/3600)

after = merged['VALUENUM_y']
estimate = e['estimated']

percent = 100*(after-estimate)/estimate

ratio = after/estimate

def remove_outlier(val, time_diff):
    val = pd.DataFrame(val)
    time_diff = pd.DataFrame(time_diff)
    # IQR
    Q1 = np.percentile(val, 25,
                    interpolation = 'midpoint')
    
    Q3 = np.percentile(val, 75,
                    interpolation = 'midpoint')
    IQR = Q3 - Q1
    
    # Upper bound
    upper = np.where(val >= (Q3+1.5*IQR))
    # Lower bound
    lower = np.where(val <= (Q1-1.5*IQR))
    val.drop(upper[0], inplace = True)
    time_diff.drop(upper[0], inplace = True)
    val.drop(lower[0], inplace = True)
    time_diff.drop(lower[0], inplace = True)
    return val, time_diff

import seaborn as sns

def plot_func(absolute, time_diff, title=''):
    plot_data = pd.concat([absolute, time_diff], axis=1)
    plot_data = plot_data.rename(columns={0:'Lab values'})
    plot_data = plot_data[plot_data['timeFromPrescription_y']>12]
    sns.regplot(x = "timeFromPrescription_y", 
            y = 'Lab values', 
            data = plot_data, 
            truncate=False)
    plt.title('Insulin<>Glucose - '+ title+ ' change in lab measurment and time taken for change')
    plt.xlabel('Time in hours')
    plt.ylabel('Glucose Levels (mg/dL)')
    plt.show()

absolute1, time_diff3 = remove_outlier(absolute, time_diff)
plot_func(absolute1, time_diff3, 'Absolute')

percent1, time_diff1 = remove_outlier(percent, time_diff)
plot_func(percent1, time_diff1, 'Percentage')

ratio1, time_diff2 = remove_outlier(ratio, time_diff)
plot_func(ratio1, time_diff2, 'Ratio')