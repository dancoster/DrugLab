from src.imputation import utils_imputation
import pandas as pd

def mask_values(df, columns, mask_rate=0.3, seed=0, logfile=None):
    """ Given DF, mask (np.nan) each columns (available) values, by mask_rate percent.
        The masking is done on avaiable values only.  """
    N_before = df.isna().sum().sum()
    masked_df = df.copy()
    mask_index = {}

    for col in columns:
        sampled_rows = masked_df[col].dropna().sample(frac=mask_rate, random_state=seed).index  # ensure random seed
        masked_df.loc[sampled_rows, col] = pd.np.nan  # mask avaiable data
        mask_index[col] = sampled_rows

    N_after = masked_df.isnull().sum().sum()
    if logfile:
        utils_imputation.write_log(logfile, f"mask_values: {N_after - N_before} values from {len(columns)} variables were masked")

    return masked_df, mask_index

def rmse(y, y_tag):
    """ @param y: column represents the real value
        @param y_tag: column represents the predicted value """
    return ((y_tag - y) ** 2).mean() ** .5

def nrmse(y, y_tag, org_max, org_min):
    """ normalize rmse by the difference between the maximum and minimum observed values.
        For the min/max we use the scale of the original observed out of all indices"""
    return rmse(y, y_tag) / (org_max - org_min)

def calculate_rmse_nrmse(df, lab_name,imputer_type,med_t_name,delta_time_after,delta_time_before,gam_func,logfile,drug_forward_type):
    df_masked, masked_ids = mask_values(df, [lab_name], mask_rate=0.1, seed=0, logfile=None)
    utils_imputation.write_log(logfile, f"\nLab: {lab_name} <> Drug: {med_t_name} - number of masked indices: {len(masked_ids[lab_name])}")
    num_masked_subjects = len(df_masked[df_masked.index.isin(masked_ids[lab_name])]['subject_id'].unique())

    # masked indices of that drug was administered at their time of admission
    masked_drug_affected_indices = list(set(df_masked[~df_masked[med_t_name].isna()].index).intersection(masked_ids[lab_name]))
    utils_imputation.write_log(logfile, f"\nLab: {lab_name} <> Drug: {med_t_name} - number of masked indices with drug admission on the same time: {len(masked_drug_affected_indices)}")
    masked_drug_affected_indices = {lab_name:masked_drug_affected_indices}
    num_subjects_who_got_drug_at_the_time_of_drug = len(df_masked[df_masked.index.isin(masked_drug_affected_indices[lab_name])]['subject_id'].unique())

    # masked indices of subjects who got drug
    indices_of_subjects_who_got_drug = df_masked[df_masked.subject_id.isin(df_masked[~df_masked[med_t_name].isna()]['subject_id'].unique())].index
    masked_indices_of_subjects_who_got_drug = list(set(indices_of_subjects_who_got_drug).intersection(masked_ids[lab_name]))
    num_masked_subjects_who_got_drug = len(df_masked[df_masked.index.isin(masked_indices_of_subjects_who_got_drug)]['subject_id'].unique())
    utils_imputation.write_log(logfile, f"\nLab: {lab_name} <> Drug: {med_t_name} - number of subjects who got the drug and there indices were masked:{num_masked_subjects_who_got_drug} with number of indices: {len(masked_indices_of_subjects_who_got_drug)}")
    masked_indices_of_subjects_who_got_drug = {lab_name:masked_indices_of_subjects_who_got_drug}

    if (imputer_type == 'drug_forward'):
        df_temp_masked_imputed = utils_imputation.drug_forward_imputation(df_masked,lab_name,med_t_name,delta_time_before,gam_func,delta_time_after,drug_forward_type)
        df_masked_imputed = df_temp_masked_imputed.groupby('subject_id')[lab_name].ffill()

    if (imputer_type == 'ffill'):
        df_masked[lab_name] = df_masked.groupby('subject_id')[lab_name].ffill()
        df_masked_imputed = df_masked

    if (imputer_type == 'linear_interpolation'):
        interpolation_func = lambda x: x.interpolate(method='linear', limit_direction='forward', axis=0)
        df[lab_name] = df.groupby('subject_id')[lab_name].transform(interpolation_func)

    # add col names
    df_masked_imputed = pd.DataFrame(df_masked_imputed, columns = df_masked.columns)

    all_rmse = calc_rmse_nrmse(df_masked_imputed,df, masked_ids,lab_name)+[len(masked_ids[lab_name]),num_masked_subjects]
    drugs_rmse = calc_rmse_nrmse(df_masked_imputed,df, masked_drug_affected_indices,lab_name)+[len(masked_drug_affected_indices[lab_name]),num_subjects_who_got_drug_at_the_time_of_drug]
    subjects_rmse = calc_rmse_nrmse(df_masked_imputed,df, masked_indices_of_subjects_who_got_drug,lab_name)+[len(masked_indices_of_subjects_who_got_drug[lab_name]),num_masked_subjects_who_got_drug]

    return(all_rmse,drugs_rmse,subjects_rmse)

def calc_rmse_nrmse(df_masked_imputed,df, masked_ids,lab_name):
    y_imputed = df_masked_imputed[df_masked_imputed.index.isin(masked_ids[lab_name])][lab_name]
    y_exist = df[df.index.isin(masked_ids[lab_name])][lab_name]
    temp_rmse = rmse(y_exist, y_imputed)

    org_min = y_exist.min()
    org_max = y_exist.max()
    temp_nrmse = nrmse(y_exist, y_imputed,org_max,org_min)

    return([temp_rmse,temp_nrmse])
