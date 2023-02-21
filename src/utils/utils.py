import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import pandas as pd
import os


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

    def __init__(self, data, res, gender="MF", age_b=0, age_a=100, ethnicity="WHITE", lab_mapping=None, load=False):
        '''
        Params
        data : path to dataset
        res : path to result output files
        gender : stratification param for gender
        age_b : stratification param for start of age group
        age_a : stratification param for end of age group
        ethnicity : stratification param for ethnicity
        lab_mapping : lab test mapping from mimic extract. Loaded externally and used in the class 
        '''
        self.data = data 
        self.res = res
        self.gender = gender
        self.age_b = age_b
        self.age_a = age_a
        self.ethnicity = ethnicity
        self.stratify_prefix = f"{age_b}-{age_a}_{gender}_{ethnicity}"

        self.res_dict_mapping_med = None
        self.d_m_l_doc = None
        if load:
            self.load_mappings()
        self.lab_mapping = LAB_MAPPING

    def load_mappings(self):
        """
        Load Medication and Lab test name mappings from MIMIC Extract and Clinically Validated sources
        """
        self.d_m_l_doc = pd.read_csv(os.path.join(self.data, "mimiciii", "1.4","preprocessed", "mapping_med_itemid_doc.csv")).drop(columns=["Unnamed: 0"])
        dict_d_m_l = self.d_m_l_doc.to_dict("records")
        self.res_dict_mapping_med = {
            v:k["Medication"] for k in dict_d_m_l for v in [int(id) for id in k["ITEMID_with_manual"][1:-1].split(",") if id != '']
        }

    def generate_med_lab_pairs(self):
        """
        Generate medication and lab test pair names.
        """
        
        d_lab_map = {k:list(v.keys()) for k, v in self.lab_mapping.items()}
        indexes = list(self.d_m_l_doc.groupby(["Medication", "lab result"]).count().index)

        med_vals = [k[0].strip() for k in indexes]
        labtest_vals = [k[1].strip() for k in indexes]
        med_vals.append('Insulin - Regular')
        labtest_vals.append('glucose')

        med_vals.append('Packed Red Blood Cells')
        labtest_vals.append('Hemoglobin')

        med_vals.append('Calcium Gluconate (CRRT)')
        labtest_vals.append('calcium')

        med_vals.append('Packed Red Blood Cells')
        labtest_vals.append('Red blood cell')

        med_vals.append('Packed Red Blood Cells')
        labtest_vals.append('Hematocrit')

        med_vals.append('Albumin')
        labtest_vals.append('Albumin')

        med_vals.append('Albumin')
        labtest_vals.append('Hematocrit')

        med_vals.append('Albumin 5%')
        labtest_vals.append('Albumin')

        med_vals.append('Albumin 5%')
        labtest_vals.append('Hematocrit')

        med_vals.append('Albumin 25%')
        labtest_vals.append('Albumin')

        med_vals.append('Albumin 25%')
        labtest_vals.append('Hematocrit')

        med_vals.append('Magnesium Sulfate')
        labtest_vals.append('Magnesium')
        l_med_lab = [(i[0], k) for i in zip(med_vals, labtest_vals) for k in d_lab_map[i[1]]]
        labtest_vals_new = [k[1] for k in l_med_lab]
        med_vals_new = [k[0] for k in l_med_lab]
        return med_vals_new, labtest_vals_new