import pandas as pd
import os
from src.utils import utils
from tqdm.notebook import tqdm

def load_data(mimic_extract_path, MIMIC_III_RAW_PATH, dups_path, data, inc_criteria_path, root_path,delta_k):
    # load mimic-Extract data
    df = pd.read_csv(mimic_extract_path)

    # remove rows with no feature values
    df = df[~df.value.isna()]

    # remove duplicated rows (the labs rows) using "Unnamed: 0" column as row_id
    dups_df = pd.read_csv(dups_path, index_col=0)
    print(df.shape)
    df = df[~df['Unnamed: 0'].isin(dups_df[dups_df['offset'] == 1]['Unnamed: 0'])]  # offset = 1 -> charttime is shifted
    print(df.shape)

    # remove duplicates because of Age/Gender bug (different ages per individual)
    # df = df.drop(columns=['Unnamed: 0', 'level_0', 'hour', 'age', 'gender', 'agegroup'], axis=1).drop_duplicates()
    df = df.drop(columns=['Unnamed: 0', 'hour', 'age', 'gender', 'agegroup'], axis=1).drop_duplicates()

    # convert charttime to datetime
    df['charttime'] = pd.to_datetime(df['charttime'])

    # remove duplicate of chartevents and labevents
    df = df.drop_duplicates(subset=['subject_id', 'hadm_id', 'icustay_id', 'charttime', 'value'], keep='first')

    # round hours
    df['charttime'] = pd.to_datetime(df['charttime'], utc=True).dt.round(freq='H')

    # Take mean per hour
    df = df.groupby(['charttime', 'subject_id', 'icustay_id', 'hadm_id', 'itemid', 'LEVEL2'], as_index=False)[
        'value'].mean().reset_index()

    # pivot table based on LEVEL 2 mapping
    df = df[['charttime', 'subject_id', 'icustay_id', 'hadm_id', 'value', 'LEVEL2']]
    df = df.pivot_table(index=['charttime', 'subject_id', 'icustay_id', 'hadm_id'], values='value',
                        columns=['LEVEL2']).reset_index()

    # Take only subset of the features
    vital_signs = ['Heart Rate', 'Respiratory rate', 'Oxygen saturation', 'Systolic blood pressure',
                   'Diastolic blood pressure',
                   'Temperature']
    labs_bmp = ['Glucose', 'Potassium', 'Sodium', 'Chloride', 'Creatinine', 'Blood urea nitrogen', 'Bicarbonate',
                'Calcium',
                'Albumin', 'Lactate dehydrogenase', 'Magnesium', 'Lactic acid']
    labs_cbc = ['Hematocrit', 'Hemoglobin', 'Platelets', 'White blood cell count', 'Red blood cell count',
                'Mean corpuscular volume', 'Lymphocytes', 'Neutrophils']
    labs_cauglation = ['Prothrombin time INR']

    df = df[['charttime', 'subject_id', 'icustay_id', 'hadm_id'] + vital_signs + labs_bmp + labs_cbc + labs_cauglation]

    # Load file of Tammy with its inclusin/exclusion criteria
    data_path = os.path.join(f"{root_path}", inc_criteria_path)
    df_data = pd.read_csv(data_path)

    # Fitler data based on Tammy's inclusin/exclusion criteria
    df = df[df.subject_id.isin(df_data.subject_id.unique())]

    # load icustays data, include first icustay (as in mimic extract)
    d_icustay = pd.read_csv(os.path.join(data, MIMIC_III_RAW_PATH, "ICUSTAYS.csv.gz"),
                            usecols=['SUBJECT_ID', 'HADM_ID', 'INTIME'])
    d_icustay['INTIME'] = pd.to_datetime(d_icustay['INTIME'])
    d_icustay = d_icustay.sort_values(by='INTIME').drop_duplicates(subset=['SUBJECT_ID'])

    # load admissions data
    d_admissions = pd.read_csv(os.path.join(data, MIMIC_III_RAW_PATH, "ADMISSIONS.csv.gz"))
    d_admissions = d_admissions[['SUBJECT_ID', 'HADM_ID', 'DEATHTIME', 'DISCHTIME', 'ADMITTIME', 'ETHNICITY']]
    d_admissions['ADMITTIME'] = pd.to_datetime(d_admissions['ADMITTIME'])
    d_admissions['DISCHTIME'] = pd.to_datetime(d_admissions['DISCHTIME'])
    #exclude admissions that DISCHTIME is before ADMITTIME
    d_admissions = d_admissions[d_admissions['ADMITTIME'] < d_admissions['DISCHTIME']]
    d_admissions = d_admissions.merge(d_icustay, on=['SUBJECT_ID', 'HADM_ID'])
    # include admissions only with icu_intime >= hosp_admittime - delta_k (delta_k = 2)
    d_admissions = d_admissions[d_admissions['INTIME'] >= (d_admissions['ADMITTIME'] - pd.Timedelta(hours=delta_k))].drop(
        columns=['INTIME'], axis=1)

    d_admissions = d_admissions.merge(df[['subject_id', 'hadm_id']].drop_duplicates(),
                                      right_on=['subject_id', 'hadm_id'], left_on=['SUBJECT_ID', 'HADM_ID']).drop(
        columns=['subject_id', 'hadm_id'], axis=1)

    # load patients data
    d_patients = pd.read_csv(os.path.join(data, MIMIC_III_RAW_PATH, "PATIENTS.csv.gz"))
    d_patients = d_patients[['SUBJECT_ID', 'DOB', 'GENDER']]

    # merge demographics (admissions and patients)
    d_demographics = d_admissions.merge(d_patients, on='SUBJECT_ID')

    # create age col
    d_demographics['age'] = round(
        pd.to_numeric((pd.to_datetime(d_demographics['ADMITTIME']) - pd.to_datetime(d_demographics['DOB'])).dt.days,
                      downcast='integer') / 365, 2)
    d_demographics = d_demographics.drop(columns=['DOB'])

    # merge df and d_demographics data
    df = df.merge(d_demographics, right_on=['SUBJECT_ID', 'HADM_ID'], left_on=['subject_id', 'hadm_id']).drop(
        columns=['SUBJECT_ID', 'HADM_ID'], axis=1)

    return (df)


def new_load_data(mimic_extract_path, MIMIC_III_RAW_PATH, data, inc_criteria_path, root_path,delta_k):
    # load mimic-Extract data
    df = pd.read_csv(mimic_extract_path)

    # remove rows with no feature values
    df = df[~df.value.isna()]

    # remove duplicate of chartevents and labevents
    df = df.sort_values(by='linksto').drop_duplicates(subset=['subject_id', 'hadm_id', 'icustay_id', 'charttime','LEVEL2','value'], keep='last')

    # remove duplicated rows with 1 hour diff
    df = fix_one_hour_bug_df(df)

    # remove duplicates because of Age/Gender bug (different ages per individual)
    df = df.drop(columns=['hour', 'age', 'gender', 'agegroup'], axis=1).drop_duplicates()

    # convert charttime to datetime
    df['charttime'] = pd.to_datetime(df['charttime'])

    # remove duplicate of chartevents and labevents
    # df = df.drop_duplicates(subset=['subject_id', 'hadm_id', 'icustay_id', 'charttime', 'value'], keep='first')

    # round hours
    df['charttime'] = pd.to_datetime(df['charttime'], utc=True).dt.round(freq='H')

    # Take mean per hour
    df = df.groupby(['charttime', 'subject_id', 'icustay_id', 'hadm_id', 'itemid', 'LEVEL2'], as_index=False)[
        'value'].mean().reset_index()

    # pivot table based on LEVEL 2 mapping
    df = df[['charttime', 'subject_id', 'icustay_id', 'hadm_id', 'value', 'LEVEL2']]
    df = df.pivot_table(index=['charttime', 'subject_id', 'icustay_id', 'hadm_id'], values='value',
                        columns=['LEVEL2']).reset_index()

    # Take only subset of the features
    vital_signs = ['Heart Rate', 'Respiratory rate', 'Oxygen saturation', 'Systolic blood pressure',
                   'Diastolic blood pressure',
                   'Temperature']
    labs_bmp = ['Glucose', 'Potassium', 'Sodium', 'Chloride', 'Creatinine', 'Blood urea nitrogen', 'Bicarbonate',
                'Calcium',
                'Albumin', 'Lactate dehydrogenase', 'Magnesium', 'Lactic acid']
    labs_cbc = ['Hematocrit', 'Hemoglobin', 'Platelets', 'White blood cell count', 'Red blood cell count',
                'Mean corpuscular volume', 'Lymphocytes', 'Neutrophils']
    labs_cauglation = ['Prothrombin time INR']

    df = df[['charttime', 'subject_id', 'icustay_id', 'hadm_id'] + vital_signs + labs_bmp + labs_cbc + labs_cauglation]

    # Load file of Tammy with its inclusin/exclusion criteria
    data_path = os.path.join(f"{root_path}", inc_criteria_path)
    df_data = pd.read_csv(data_path)

    # Fitler data based on Tammy's inclusin/exclusion criteria
    df = df[df.subject_id.isin(df_data.subject_id.unique())]

    # load icustays data, include first icustay (as in mimic extract)
    d_icustay = pd.read_csv(os.path.join(data, MIMIC_III_RAW_PATH, "ICUSTAYS.csv.gz"),
                            usecols=['SUBJECT_ID', 'HADM_ID', 'INTIME'])
    d_icustay['INTIME'] = pd.to_datetime(d_icustay['INTIME'])
    d_icustay = d_icustay.sort_values(by='INTIME').drop_duplicates(subset=['SUBJECT_ID'])

    # load admissions data
    d_admissions = pd.read_csv(os.path.join(data, MIMIC_III_RAW_PATH, "ADMISSIONS.csv.gz"))
    d_admissions = d_admissions[['SUBJECT_ID', 'HADM_ID', 'DEATHTIME', 'DISCHTIME', 'ADMITTIME', 'ETHNICITY']]
    d_admissions['ADMITTIME'] = pd.to_datetime(d_admissions['ADMITTIME'])
    d_admissions['DISCHTIME'] = pd.to_datetime(d_admissions['DISCHTIME'])
    #exclude admissions that DISCHTIME is before ADMITTIME
    d_admissions = d_admissions[d_admissions['ADMITTIME'] < d_admissions['DISCHTIME']]
    d_admissions = d_admissions.merge(d_icustay, on=['SUBJECT_ID', 'HADM_ID'])
    # include admissions only with icu_intime >= hosp_admittime - delta_k (delta_k = 2)
    d_admissions = d_admissions[d_admissions['INTIME'] >= (d_admissions['ADMITTIME'] - pd.Timedelta(hours=delta_k))].drop(
        columns=['INTIME'], axis=1)

    d_admissions = d_admissions.merge(df[['subject_id', 'hadm_id']].drop_duplicates(),
                                      right_on=['subject_id', 'hadm_id'], left_on=['SUBJECT_ID', 'HADM_ID']).drop(
        columns=['subject_id', 'hadm_id'], axis=1)

    # load patients data
    d_patients = pd.read_csv(os.path.join(data, MIMIC_III_RAW_PATH, "PATIENTS.csv.gz"))
    d_patients = d_patients[['SUBJECT_ID', 'DOB', 'GENDER']]

    # merge demographics (admissions and patients)
    d_demographics = d_admissions.merge(d_patients, on='SUBJECT_ID')

    # create age col
    d_demographics['age'] = round(
        pd.to_numeric((pd.to_datetime(d_demographics['ADMITTIME']) - pd.to_datetime(d_demographics['DOB'])).dt.days,
                      downcast='integer') / 365, 2)
    d_demographics = d_demographics.drop(columns=['DOB'])

    # merge df and d_demographics data
    df = df.merge(d_demographics, right_on=['SUBJECT_ID', 'HADM_ID'], left_on=['subject_id', 'hadm_id']).drop(
        columns=['SUBJECT_ID', 'HADM_ID'], axis=1)

    return (df)

def fix_one_hour_bug_df(df):
    #rename cols to fit func fix_one_hour_bug, and sort data by time
    df_x = df.rename(columns={"linksto": "TABLE"})
    df_x['TABLE'] = df_x['TABLE'].fillna('labevents')
    df_x.columns = [x.upper() for x in df_x.columns]
    df_x = df_x.sort_values(by=['SUBJECT_ID', 'CHARTTIME'])
    df_x['OLD_ITEMID'] = df_x['ITEMID']
    df_x['ITEMID'] = df_x.LEVEL2

    df_x['VALUE'] = round(df_x['VALUE'],1)
    df_x = utils.fix_one_hour_bug(df_x)

    df = df_x.copy()
    df.columns = [x.lower() for x in df.columns]
    df = df.rename(columns={"level1": "LEVEL1", "level2": "LEVEL2"})
    df['itemid'] = df['old_itemid']
    df = df.drop(columns=['old_itemid'])

    return(df)

def inflate_hourly_frequency_obs_per_id(df_pat):
    new_df_pat = df_pat.asfreq('H')
    new_df_pat['charttime'] = new_df_pat.index
    new_df_pat['subject_id'] = new_df_pat['subject_id'].ffill()
    new_df_pat = new_df_pat.reset_index(drop=True)

    return(new_df_pat)

def inflate_hourly_frequency_obs(df_data):
    new_df_data = pd.DataFrame(columns=df_data.columns)
    for s_id in tqdm(df_data.subject_id.unique(), mininterval=30):
        df_pat = df_data[df_data.subject_id == s_id]

        #convert chartime to index
        df_pat.charttime= pd.to_datetime(df_pat.charttime)
        df_pat.index = df_pat.charttime
        new_df_pat = inflate_hourly_frequency_obs_per_id(df_pat)
        new_df_data = pd.concat([new_df_data,new_df_pat], axis =0)

    return (new_df_data)
