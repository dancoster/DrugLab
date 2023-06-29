import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import pandas as pd
import os
from sklearn import linear_model
import json
from src.utils import constants

# Util Functions

def sort_rows_with_time(p_corrs, s_corrs, after_windows):
    s_p = sorted([k for k in zip(p_corrs, after_windows)], key=lambda k: k[1][0])
    p_corrs = [k[0][0] for k in s_p]
    after_windows1 = [str(k[1]) for k in s_p]
    s_s = sorted([k for k in zip(s_corrs, after_windows)], key=lambda k: k[1][0])
    s_corrs = [k[0][0] for k in s_s]
    after_windows2 = [str(k[1]) for k in s_p]
    return p_corrs, s_corrs, after_windows1, after_windows2

def plot_corrs(corrs, after_windows, after_windows_map, ax, title='', plot_name='',  after_window_info=None):

    p_corrs = [c[0] for c in corrs]
    s_corrs = [c[1] for c in corrs]
    final_plot_name = f'{plot_name}_{title}'
    after_windows = [after_windows_map[a.split("_")[-2]] for a in after_windows]
    
    p_corrs, s_corrs, after_windows1, after_windows2 = sort_rows_with_time(p_corrs, s_corrs, after_windows)

    ax[0].plot(after_windows1, p_corrs, '-o')
    ax[0].set_title(f'{final_plot_name} Pearsons Corr')
    ax[0].set(xlabel='Time (h)', ylabel='Correlation')
    ax[0].set_xticks(after_windows1)
    ax[0].grid()

    ax[1].plot(after_windows2, s_corrs, '-o')
    ax[1].set_title(f'{final_plot_name} Spearmans Corr')
    ax[1].set(xlabel='Time (h)', ylabel='Correlation')
    ax[1].set_xticks(after_windows2)
    ax[1].grid()
    

def change_col_to_datetime(inputevents_mv, feature):
    """

    Args:
        inputevents_mv (pd.DataFrame): _description_
        feature (str): _description_

    Returns:
        pd.DataFrame: 
    """
    inputevents_mv[feature] = pd.to_datetime(inputevents_mv[feature])
    return inputevents_mv

def plot_func(lab, presc, d, dirname, plot_dir, plot_dir1, window=(1,24), title='', unit='', labels=None, plot_name='', ax=None):    
    plot_data = d
    if ax is None:
        sns.regplot(x = "time", 
                y = 'data', 
                data = plot_data.sort_values(["time"]), 
                truncate=False)
        n = plot_data.shape[0]
        plt.title(lab+'<>'+presc+'- '+ title+ ' \nchange in lab measurment and time taken for change')
        plt.xlabel('Time (h)')
        plt.ylabel(f"{title} change in {lab} lab measurment ({unit})")
        if labels is not None:
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            plt.legend([extra for i in range(5)], (f'Pearson Correlation = {round(labels[0][0], 4)}', f'Pearson Correlation p-value = {round(labels[0][1], 4)}', f'Spearmans Correlation = {round(labels[1][0], 4)}', f'Spearmans Correlation p-value = {round(labels[1][1], 4)}', f'Number of data points = {n}'))
        if not os.path.isdir(os.path.join(plot_dir1, f"{lab}<>{presc}")):
            os.mkdir(os.path.join(plot_dir1, f"{lab}<>{presc}"))
        if dirname is None or dirname == "":
            plt.savefig(os.path.join(plot_dir1, f"{lab}<>{presc}", plot_name+".png"))
        else:
            if not os.path.isdir(os.path.join(plot_dir1, f"{lab}<>{presc}", dirname)):
                os.mkdir(os.path.join(plot_dir1, f"{lab}<>{presc}", dirname))
            plt.savefig(os.path.join(plot_dir1, f"{lab}<>{presc}", dirname, plot_name+".png"))
        plt.clf()
    
    else:
        sns.regplot(
                ax=ax,
                x = "time", 
                y = 'data', 
                data = plot_data.sort_values(["time"]), 
                truncate=False)
        n = plot_data.shape[0]
        ax.set_title(lab+'<>'+presc+'- '+ title+ ' \nchange in lab measurment and time taken for change')
        ax.set(xlabel='Time (h)', ylabel=f"{title} change in {lab} lab measurment ({unit})")
        ax.grid()
        if labels is not None:
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            ax.legend([extra for i in range(5)], (f'Pearson Correlation = {round(labels[0][0], 4)}', f'Pearson Correlation p-value = {round(labels[0][1], 4)}', f'Spearmans Correlation = {round(labels[1][0], 4)}', f'Spearmans Correlation p-value = {round(labels[1][1], 4)}', f'Number of data points = {n}'))

def remove_outlier(val, time_diff):
    val = pd.DataFrame(val)
    time_diff = pd.DataFrame(time_diff)
    
    # IQR
    Q1 = np.percentile(val, 25, method = 'midpoint')        
    Q3 = np.percentile(val, 75, method = 'midpoint')
    IQR = Q3 - Q1        
    
    # Upper bound
    upper = np.where(val >= (Q3+1.5*IQR))
    # Lower bound
    lower = np.where(val <= (Q1-1.5*IQR))

    # Filtering
    if len(upper) > 0:
        val.drop(upper[0], inplace = True)
        time_diff.drop(upper[0], inplace = True)
    if len(lower) > 0:
        val.drop(lower[0], inplace = True)
        time_diff.drop(lower[0], inplace = True)
    return val, time_diff

def check_med2(row):
    if row["HADM_ID"] in t_med2["HADM_ID"].to_list():
        if row["ITEMID"] in t_med2[t_med2["HADM_ID"]==row["HADM_ID"]]["ITEMID"].to_list():
            return True
    return False

def get_med2(row):
    temp = t_med2[t_med2["HADM_ID"]==row["HADM_ID"]] 
    return temp[temp["ITEMID"]==row["ITEMID"]].iloc[0]


def get_normalized_trend(data):
    selected = data[['VALUENUM', 'hours_from_med']]
    if selected.shape[0]<2:
        return float("NaN")
    reg = linear_model.LinearRegression()
    reg.fit(np.array(data['hours_from_med']).reshape(-1,1), np.array(data['VALUENUM']).reshape(-1,1))
    return reg.coef_[0][0]

def get_normalized_trend_np(data):
    selected = data[['VALUENUM', 'hours_in']]
    print(selected)
    if selected.shape[0]<2:
        return float("NaN")
    print(np.array(data['hours_from_med']), np.array(data['VALUENUM']))
    t = np.polyfit(np.array(data['hours_from_med']), np.array(data['VALUENUM']), 1,full=True)
    coefficients, residuals, _, _, _ = t
    print(t)
    mse = residuals[0]/(len(selected.index))
    nrmse = np.sqrt(mse)/(selected.max() - selected.min())
    return 1

# Utils class
class AnalysisUtils:

    def __init__(self, data, res, gender="MF", age_b=0, age_a=100, ethnicity="WHITE", load="MANUAL_MAPPING_MIMIC", lab_mapping=constants.LAB_MAPPING, top_k_meds=30):
        '''
        Params
        data : path to dataset
        res : path to result output files
        gender : stratification param for gender
        age_b : stratification param for start of age group
        age_a : stratification param for end of age group
        ethnicity : stratification param for ethnicity
        load: load mappings for lab test and medication data - "MANUAL_MAPPING" (labtest and medication names taken from constants.py), "AUTOMATIC_MAPPING" (based on number of subjects associated with the medication/labtest)
        lab_mapping : lab test mapping from mimic extract. Loaded externally and used in the class 
        top_k_meds : top k meds based on number of subjects is chosen
        '''
        # Paths to raw, preprocessed dataset files and analysis result files
        self.data = data 
        self.res = res
        
        # Stratification parmas
        self.gender = gender
        self.age_b = age_b
        self.age_a = age_a
        self.ethnicity = ethnicity
        self.stratify_prefix = f"{age_b}-{age_a}_{gender}_{ethnicity}"
        
        # Mappings
        self.load_mappings(type=load, lab_mapping=lab_mapping, num_meds=top_k_meds)    

    def load_mappings(self, type, lab_mapping=None, num_meds=30):
        """
        Load Medication and Lab test name mappings from MIMIC Extract and Clinically Validated sources
        """
        if type=="MANUAL_MAPPING_MIMIC":
            self.med_mapping = constants.MIMIC_DICT_MED_MAPPING
            self.lab_mapping = lab_mapping if lab_mapping is not None else constants.LAB_MAPPING
        elif type=="AUTOMATIC_MAPPING_MIMIC":
            # read data
            inputevents_mv = pd.read_csv(os.path.join(self.data, constants.MIMIC_III_RAW_PATH, "INPUTEVENTS_MV.csv.gz"))
            med_data = pd.read_csv(os.path.join(self.data, constants.MIMIC_III_RAW_PATH, "D_ITEMS.csv.gz"))
            inputevents_mv = pd.merge(inputevents_mv, med_data, how="inner", on="ITEMID")
            # get top k med and generate mapping
            self.med_mapping = {k[0]:k[1] for k in inputevents_mv.groupby(["ITEMID", "LABEL", "HADM_ID"]).count().reset_index()[["ITEMID", "LABEL", "HADM_ID"]].groupby(["ITEMID", "LABEL"]).count().sort_values("HADM_ID", ascending=False).head(num_meds).to_dict()["HADM_ID"].keys()}
            self.lab_mapping = lab_mapping if lab_mapping is not None else constants.LAB_MAPPING
        elif type=="MANUAL_MAPPING_HIRID":
            self.med_mapping = None
            self.lab_mapping = lab_mapping if lab_mapping is not None else constants.HIRID_LAB_IDS
        elif type=="AUTOMATIC_MAPPING_HIRID":
            # read data
            pharma_records_paths = [i for iq, i in enumerate(os.walk(os.path.join(self.data, "pharma_records"))) if iq==1][0][2]
            pharma_records = pd.concat([pd.read_csv(os.path.join(self.data, "pharma_records", 'csv', file)) for file in pharma_records_paths])
            pharma_records = pharma_records.rename(columns={"pharmaid":"variableid"})
            
            g_table = pd.read_csv(os.path.join(self.res, 'general_table.csv'))
            h_var_ref = pd.read_csv(os.path.join(self.res, 'hirid_variable_reference.csv')).rename(columns={"ID":"variableid"})
        
            pharma_records_with_name = pd.merge(pharma_records, h_var_ref, on="variableid", how="inner")
            pharma_records_with_name = pd.merge(pharma_records_with_name, g_table, on="patientid", how="inner")
            pharma_records_with_name = pharma_records_with_name.rename(columns={
                "givenat":"STARTTIME",
                "admissiontime":"ADMITTIME",
                "enteredentryat":"ENDTIME",
                "variableid":"ITEMID",
                "patientid":"HADM_ID",
                "Variable Name":"LABEL",
                "age":"AGE",
                "sex":"GENDER",
            })
            # get top k med and generate mapping
            self.med_mapping = {k[0]:k[1] for k in pharma_records_with_name.groupby(["ITEMID", "LABEL", "HADM_ID"]).count().reset_index()[["ITEMID", "LABEL", "HADM_ID"]].groupby(["ITEMID", "LABEL"]).count().sort_values("HADM_ID", ascending=False).head(num_meds).to_dict()["HADM_ID"].keys()}
            self.lab_mapping = lab_mapping if lab_mapping is not None else constants.HIRID_LAB_IDS
        else:
            print("No mapping type chosen. Running on all labtests and medications")
            return

    def generate_med_lab_pairs(self):
        """
        Generate medication and lab test pair names.
        """
        med_vals = [k[0] for k in constants.MIMIC_III_MED_LAB_PAIRS]
        labtest_vals = [k[1] for k in constants.MIMIC_III_MED_LAB_PAIRS]
        return med_vals, labtest_vals

def countna(df, features, pivoted_data=True, feature_col_name=None, count_patients=False):
    """ Counts null values per feature in features.
        If df is not pivoted, the function can also count how many patients had null values per feature.
    """
    if count_patients and not pivoted_data:
        df = df[df["Value"].isna()]
        df = df.groupby([feature_col_name])["Patient ID"].nunique()

    else:
        if pivoted_data:
            df = df[features].isnull().sum()

        else:
            df["Value"] = df["Value"].isnull().astype(int)
            df = df.groupby([feature_col_name])["Value"].sum()

    df = pd.DataFrame(df)

    return df

def count_size(df, features, pivoted_data=True, feature_col_name=None, count_patients=False):
    """ Counts non-null values per feature in features.
        If df is not pivoted, the function can also count how many patients had values per feature.
    """

    # Count patients
    if count_patients:
        if pivoted_data:
            df = df[features].mask(df[features].isna(), df["Patient ID"], axis=1)
            df = df[features].nunique()

        else:
            df = df[df["Value"].notna()]
            df = df.groupby([feature_col_name])["Patient ID"].nunique()


    # Count values
    else:
        if pivoted_data:
            df = df[features].notna().sum()

        else:
            df["Value"] = df["Value"].notna().astype(int)
            df = df.groupby([feature_col_name])["Value"].sum()

    df = pd.DataFrame(df)
    return df


def remove_inhuman_values(df, path_for_ranges, pivoted_data=True, feature_col_name=None, df_name= "merged"):
    """ Masks values that exceed the possible known range.
        Works on data both before and after pivot, by pivoted_data flag.
        'feature_col_name' is the name of the features column in the non-pivoted format.
    """
    with open(path_for_ranges) as json_ranges:
        ranges = json.load(json_ranges)

    features, missing_features = [], []
    unique_features = df.columns if pivoted_data else df[feature_col_name].unique()
    for x in unique_features:
        features.append(x) if x in ranges.keys() else missing_features.append(x)

    # Count nulls for statistics
    val_sizes = count_size(df.copy(), features, pivoted_data, feature_col_name)
    patients_sizes = count_size(df.copy(), features, pivoted_data, feature_col_name, count_patients=True)
    sizes = (val_sizes, patients_sizes)
    nones_before = countna(df.copy(), features, pivoted_data, feature_col_name)
    if not pivoted_data:
        pts_before = countna(df.copy(), features, pivoted_data, feature_col_name, count_patients=True)

    if pivoted_data:
        df[features] = df[features].apply(lambda c: c.mask(~c.between(ranges[c.name]['min'], ranges[c.name]['max'])))
    else:
        for feature in features:
            mask_cond = (df[feature_col_name] == feature) & \
                        (~df["Value"].between(ranges[feature]['min'], ranges[feature]['max']))
            df["Value"] = df.mask(mask_cond)["Value"]

    if missing_features:
        print(f"The following features don't have ranges in the ranges path specified, "
              f"and therefore ignored:\n{missing_features}")

    if pivoted_data:
        inhuman_statistics(df, features, ranges, nones_before, sizes, None, pivoted_data, feature_col_name,
                           df_name=df_name)
    else:
        inhuman_statistics(df, features, ranges, nones_before, sizes, pts_before, pivoted_data,
                           feature_col_name, df_name=df_name)

    return df

def inhuman_statistics(df, features, ranges, nones_before, sizes, pts_before=None, pivoted_data=True,
                       feature_col_name=None, df_name= "merged"):
    """ Displays percentage of inhuman values that were removed.
        If non-pivoted, also shows percentage of patients.
    """
    # N, Patients, min human range, max human range
    stats = pd.merge(sizes[0], sizes[1],left_index=True, right_index=True, how="outer")
    stats.columns = ["N", "Patients"]
    stats = pd.merge(stats, pd.DataFrame.from_dict(ranges).T, how="outer", right_index=True, left_index=True)

    # Inhuman removed
    nones_after = countna(df.copy(), features, pivoted_data, feature_col_name)
    removed_values = pd.merge(pd.merge(nones_after, nones_before, left_index=True, right_index=True), sizes[0], left_index=True, right_index=True)

    if pivoted_data:
        removed_values = 100 * (removed_values["0_x"] - removed_values["0_y"]) / removed_values[0]
    else:
        removed_values = 100 * (removed_values["Value_x"] - removed_values["Value_y"])/ removed_values["Value"]

    removed_values = removed_values[removed_values != 0].rename("Inhuman Values(%)")
    stats = pd.merge(stats, removed_values, left_index=True, right_index=True, how="left")

    # Patients with removed
    if not pivoted_data:
        pts_after = countna(df.copy(), features, pivoted_data, feature_col_name, count_patients=True)
        if pts_before.sum().sum():
            patients_with_removed = 100 * (pts_after.iloc[:,0] - pts_before.iloc[:,0])/(sizes[1].iloc[:,0])
        else:
            patients_with_removed = 100 * pts_after.iloc[:,0] /(sizes[1].iloc[:,0])
        stats = pd.merge(stats, patients_with_removed, how="left", left_index=True, right_index=True)

    if pivoted_data:
        print(f"In total, {(nones_after.sum() - nones_before.sum()).sum()} invalid values were removed:")
    else:
        pts_with_nones = df[df["Value"].isna()]["Patient ID"].unique()
        print(f"In total, {(nones_after.sum() - nones_before.sum()).sum()} invalid values from {len(pts_with_nones)} patients were removed:")

    stats = pd.DataFrame(stats.round(3))
    stats.to_csv(os.path.join(f"inhuman_statistics_{df_name}.csv"))
    stats = stats.dropna(how="any")

    return(stats)


def convert_units_features(m_labs):
    df = m_labs.copy()
    #convert Farnhiet to Celsius and change ITEMID name
    df[(df.ITEMID == 'Temperature (F)')]['VALUENUM'] = (df[(df.ITEMID == 'Temperature (F)')]['VALUENUM'] - 32)*(5/9)
    df[(df.ITEMID == 'Temperature (F)')]['ITEMID'] ='Temperature (C)'

    return(df)

def remove_and_count_inhuman_values(m_labs):
    path_for_ranges = 'feature_ranges.json' # json of inhuman values
    pivoted_data = False # row per each value of clinical measures
    feature_col_name = 'ITEMID'
    df_name = 'mimic_inhuman' #file name
    df = m_labs

    #rename cols to fit 'remove_inhuman_values' function
    df = df.rename(columns={"VALUENUM": "Value", "HADM_ID":"Patient ID"})
    df_in = remove_inhuman_values(df, path_for_ranges, pivoted_data=False, feature_col_name='ITEMID', df_name= "merged")

    #change cols to oringinal names
    df_in = df_in.rename(columns={"Value":"VALUENUM","Patient ID":"HADM_ID"})

    return df_in
