import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import pandas as pd
import os

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
from sklearn import datasets, linear_model, metrics

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
            # get top k and generate data
            self.med_mapping = {k[0]:k[1] for k in inputevents_mv.groupby(["ITEMID", "LABEL", "HADM_ID"]).count().reset_index()[["ITEMID", "LABEL", "HADM_ID"]].groupby(["ITEMID", "LABEL"]).count().sort_values("HADM_ID", ascending=False).head(num_meds).to_dict()["HADM_ID"].keys()}
            self.lab_mapping = lab_mapping if lab_mapping is not None else constants.LAB_MAPPING
        elif type=="MANUAL_MAPPING_HIRID":
            self.med_mapping = None
            self.lab_mapping = lab_mapping if lab_mapping is not None else constants.HIRID_LAB_IDS
        elif type=="AUTOMATIC_MAPPING_HIRID":
            self.med_mapping = None
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
