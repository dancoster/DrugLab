import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import win32com.client
from tqdm import tqdm
from src.imputation import utils_general
from paths import config
# Load configuration from JSON file

# Paths
root_path = config['paths']['root_path']

def setup_io_config(root_path):
    """
    Input - Output config. Add dataset paths
    :root_path -> Repo path which contains 'data' and 'res' folders
    """

    # MIMIC
    is_shortcut = True if "data.lnk" in os.listdir(root_path) else False

    if (is_shortcut):
        path_shortcut = os.path.join(root_path, "data.lnk")
        shell = win32com.client.Dispatch("WScript.Shell")
        mimic_data = shell.CreateShortCut(path_shortcut).Targetpath
    else:
        mimic_data = os.path.join(f"{root_path}", "data")
    mimic_path = os.path.join(f"{root_path}", "results")

    # HIRID

    hirid_data = f'{root_path}/data/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/'
    hirid_path = f'{root_path}/data/hirid-a-high-time-resolution-icu-dataset-1.1.1'

    return mimic_data, mimic_path, hirid_data, hirid_path

data, res, raw_path, res_path = setup_io_config(root_path=root_path)

mimic_extract_path = config['paths']['mimic_extract_path']
inc_criteria_path = config['paths']['inc_criteria_path']
MIMIC_III_RAW_PATH = config['paths']['MIMIC_III_RAW_PATH']
dups_path = config['paths']['dups_path']
final_complete_path = config['paths']['final_complete']
df_meds_path = config['paths']['df_meds']
df_not_imputed_path = config['paths']['df_not_imputed']
reports_path = config['paths']['reports_path']
df_d_items_path = os.path.join(data, config['paths']['df_d_items'])
inputevents_mv_path = os.path.join(data, config['paths']['inputevents_mv'])
df_inhuman_path = config['paths']['df_inhuman']

# Parameters
min_p_val = config['parameters']['min_p_val']
min_no_patients = config['parameters']['min_no_patients']
n_patients = config['parameters']['n_patients']
n_folds = config['parameters']['n_folds']
run_standardization = config['parameters']['run_standardization']
imputation_methods = config['parameters']['imputation_methods']
drug_forward_params = config['parameters']['drug_forward_params']
models_vector = config['models_vector']

# Features
vital_signs = config['features']['vital_signs']
labs_bmp = config['features']['labs_bmp']
labs_cbc = config['features']['labs_cbc']
labs_cauglation = config['features']['labs_cauglation']
extra_features = config['features']['extra_features']

features = vital_signs + labs_bmp + labs_cbc + labs_cauglation + extra_features
# Merged Features
# Load dataframes
final_complete = pd.read_csv(final_complete_path)
final_complete.loc[final_complete.LAB_NAME == 'Platelets', 'LAB_NAME'] = 'Platelets count'
df_meds = pd.read_csv(df_meds_path, index_col=[0])
df_meds = df_meds[df_meds['No. of Patients'] > min_no_patients]
df_meds = df_meds[df_meds['bon_corrected'] < min_p_val]
df_meds = df_meds[df_meds['Lab Name'].isin(config['features']['vital_signs'] + config['features']['labs_bmp'] + config['features']['labs_cbc'] + config['features']['labs_cauglation'])]

# Further processing
df = pd.read_csv(df_not_imputed_path)
df = df.rename(columns={'Red blood cell count':'Red blood cell'})
old_df = df.copy()

# Subsampling
subsample_ids = np.random.choice(old_df.subject_id.unique(), size=n_patients, replace=False)
df = old_df[old_df.subject_id.isin(subsample_ids)]

# Additional processing steps would go here
# Add target
df['GENDER'] = df['GENDER'].map({'M': True, 'F': False})
df['time_to_death'] = (pd.to_datetime(df['DEATHTIME'], utc=True) - pd.to_datetime(df['charttime']))
df['time_to_death'] = round(df['time_to_death'] / np.timedelta64(1, 'h'), 2)
df['target'] = df['time_to_death'] < 48
df = df.drop(columns=['time_to_death', 'icustay_id', 'hadm_id', 'ADMITTIME', 'DEATHTIME', 'ETHNICITY', 'DISCHTIME'], axis=1)

# Data Processing: Inflation
df_data = df.copy()
df_data['charttime'] = pd.to_datetime(df_data['charttime'], utc=True)

patient_ids = df_data.subject_id.unique()
subsample_ids = np.random.choice(patient_ids, size=n_patients, replace=False)
df_data = df_data[df_data.subject_id.isin(subsample_ids)]
df_data.rename(columns={'Platelets': 'Platelets count'}, inplace=True)

df_data = df_data[['subject_id', 'charttime'] + vital_signs + labs_bmp + labs_cbc + labs_cauglation]

# Inflate data based on configurable method
inflation_method = getattr(utils_general, config['processing']['inflation_method'])
time_index_column = config['processing']['time_index_column']

df_final = df.iloc[:,df.columns.isin(features+['age','GENDER','charttime','subject_id','target'])]
df_final['charttime'] = pd.to_datetime(df_final['charttime'], utc=True)

new_df_data = pd.DataFrame(columns=df_data.columns)
for s_id in tqdm(df_data.subject_id.unique()):
    df_pat = df_data[df_data.subject_id == s_id]
    df_pat.charttime= pd.to_datetime(df_pat.charttime)
    df_pat.index = df_pat.charttime
    new_df_pat = utils_general.inflate_hourly_frequency_obs(df_pat)
    new_df_data = pd.concat([new_df_data,new_df_pat], axis =0)

df_data = new_df_data.copy().reset_index(drop=True)

# Example of loading drug_forward_params
drug_forward_params.update({
    'mimic_data_querier': final_complete,
    'inputevents_mv': pd.read_csv(inputevents_mv_path),
    'df_d_items': pd.read_csv(df_d_items_path),
    'df_inhuman': pd.read_csv(df_inhuman_path, encoding= 'unicode_escape'),
    'temp_df_meds': df_meds.rename(columns={"lab_name": "Lab Name", 'med_name':'Med Name'})
})

