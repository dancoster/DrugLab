from src.parsers import mimic, hirid
from src.modeling import discovery, plots, querier
from src.utils import constants
import sys
# import win32com.client
import os
import pandas as pd

def setup_io_config(root_path, data_path=None):
    """
    Input - Output config. Add dataset paths
    :root_path -> Repo path which contains 'data' and 'res' folders
    """

    # MIMIC
    is_shortcut = True if "data.lnk" in os.listdir(root_path) else False 
    
    if (is_shortcut):
        path_shortcut =  os.path.join(root_path, "data.lnk")
        shell = win32com.client.Dispatch("WScript.Shell")
        mimic_data = shell.CreateShortCut(path_shortcut).Targetpath
    else:
        if data_path is None:
            mimic_data = os.path.join(f"{root_path}", "data")
        else:
            mimic_data = data_path
    mimic_path = os.path.join(f"{root_path}", "results")

    # HIRID
    hirid_data = f'{data_path}/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/'
    hirid_path = f'{root_path}/results/hirid/'

    return mimic_data, mimic_path, hirid_data, hirid_path

def setup_stratification_config():
    gender="MF"
    age_b=40
    age_a=80
    ethnicity="WHITE"
    lab_mapping= constants.LAB_MAPPING
    b_w = [(0,6), (6,12)]
    a_w = [(0,4), (4,8), (8,12)]
    before_windows = b_w
    after_windows = a_w
    return gender, age_a, age_b, ethnicity, lab_mapping, before_windows, after_windows
# IO Config
# root_path ="C:\\Users\\danco\\My Drive\\Master\\Datasets\\MIMIC iii"
# root_path = "/Users/pavan/Library/CloudStorage/GoogleDrive-f20190038@hyderabad.bits-pilani.ac.in/My Drive/TAU/Code/DrugLab"
root_path = "/home/gaga/yannam/DrugLab"
data_path = "/home/gaga/data/physionet/HiRiD"
data, res, raw_path, res_path = setup_io_config(root_path=root_path, data_path=data_path)

# Stratification Config
gender, age_a, age_b, ethnicity, lab_mapping, before_windows, after_windows = setup_stratification_config()

# import ast
# temp = pd.read_csv(os.path.join(res_path, 'before_after_windows_main_med_lab_first_val_40-80_MF_WHITE_doc_eval_new_win_(0, 10).csv')).drop(columns=["Unnamed: 0"])
# col_vals = []
# all_types = set(["abs", "mean", "std", "trends", "time"])
# cols_b = [f"before_{t}_{b_w}" for b_w in before_windows for t in all_types]
# cols_a = [f"after_{t}_{a_w}" for a_w in after_windows for t in all_types]
# cols = cols_b.copy()
# cols.extend(cols_a)
# for col in cols:
#     col_vals.append(
#         temp.assign(dict=temp[col].dropna().map(lambda d:ast.literal_eval(d).items())).explode("dict", ignore_index=True).assign(
#             LAB_ITEMID=lambda df: df.dict.str.get(0),
#             temp=lambda df: df.dict.str.get(1)
#         ).drop(columns=["dict"]+cols).astype({'temp':'float64'}).rename(columns={"temp":f"{col}_sp"}).dropna(subset=["LAB_ITEMID"])
#     )
# for i in range(1, len(col_vals)):
#     col_vals[i] = pd.merge(col_vals[i-1], col_vals[i], how="outer", on=list(set(temp.columns).difference(cols))+["LAB_ITEMID"])
# final = col_vals[-1][list(set(temp.columns).difference(cols))+["LAB_ITEMID"]+[f"{col}_sp" for col in cols]]
# final["LAB_NAME"] = final["LAB_ITEMID"]
# final = final.rename(columns={"ITEMID":"MED_NAME"})
# final.to_csv(os.path.join(res_path, f"before_after_windows_main_med_lab_trends_first_val_40-80_MF_WHITE_doc_eval_win_(0, 10).csv"))

# Med lab pairs
med_lab_pair_paths = ['before_after_windows_main_med_lab_trends_first_val_40-80_MF_WHITE_doc_eval_win_(0, 10).csv', 'before_after_windows_main_med_lab_trends_first_val_40-80_MF_WHITE_doc_eval_win_(10, 20).csv', 'before_after_windows_main_med_lab_trends_first_val_40-80_MF_WHITE_doc_eval_win_(20, 30).csv', 'before_after_windows_main_med_lab_trends_first_val_40-80_MF_WHITE_doc_eval_win_(30, 250).csv']
med_lab_pairs = pd.concat([pd.read_csv(os.path.join(res_path, path)).drop(columns=["Unnamed: 0"]) for path in med_lab_pair_paths])
print(med_lab_pairs.columns)
print(med_lab_pairs.iloc[0])

## Discovery Analysis for the queried medication and lab test pairs in the chosen before and after windows
analyzer = discovery.ClinicalDiscoveryAnalysis(med_lab_pairs)
types_l = ["abs", "mean", "std", "trends"]
pvals_med_lab = analyzer.analyze(before_windows, after_windows, min_patients=100, types_l=types_l)
pvals_med_lab.to_csv(os.path.join(res_path, "hirid_pvals_s.csv"))
pval, hard, bonferroni, fdr = analyzer.generate_significant(pvals_med_lab.dropna(subset=["TTest Paired"]))

lab_parts = "non_vital_signs"
fdr.to_csv(os.path.join(res_path, f"hirid_sig_med_lab_pairs_fdr_{lab_parts}.csv"))
bonferroni.to_csv(os.path.join(res_path, f"hirid_sig_med_lab_pairs_bonferroni_{lab_parts}.csv"))
hard.to_csv(os.path.join(res_path, f"hirid_sig_med_lab_pairs_hard_{lab_parts}.csv"))
pval.to_csv(os.path.join(res_path, f"hirid_sig_med_lab_pairs_pval_{lab_parts}.csv"))

from scipy import stats
import numpy as np

for b_w in before_windows:
    for a_w in after_windows:
        print(b_w, a_w)
        med_lab_pairs[f"ratio_{b_w}_{a_w}"] = med_lab_pairs[f"after_abs_{a_w}_sp"] / med_lab_pairs[f"before_abs_{b_w}_sp"]

pairs = set(med_lab_pairs.set_index(["Med Name", "Lab Name"]).index)
discovery_res2 = []
for med_name, lab_name in pairs:
    stat_test_df = []
    vals1 = med_lab_pairs[med_lab_pairs["LAB_NAME"]==lab_name]
    vals1 = vals1[vals1["MED_NAME"]==med_name]
    for a_w in after_windows:
        for b_w in before_windows:
            vals = vals1[f"ratio_{b_w}_{a_w}"].replace([np.inf, -np.inf], np.nan).dropna()
            if vals.shape[0]>100:
                res = stats.ttest_1samp(vals.to_numpy(), popmean=1)
                row = {
                    "Med Name": med_name,
                    "Lab Name": lab_name,
                    "Before Window (in Hours)": b_w,
                    "After Window (in Hours)": a_w,
                    "No. of Patients": vals.shape[0],
                    "1-Sampled Ttest" : res.pvalue
                }
            stat_test_df.append(row)
    if len(stat_test_df)>0:
        discovery_res2.extend(stat_test_df)
res_df2 = pd.DataFrame(discovery_res2)

analyzer = discovery.ClinicalDiscoveryAnalysis([])
pval, hard, bonferroni, fdr = analyzer.generate_significant(res_df2.dropna(subset=["1-Sampled Ttest"]))

lab_parts = "ratio_non_vital_signs"
fdr.to_csv(os.path.join(res_path, f"hirid_sig_med_lab_pairs_fdr_{lab_parts}.csv"))
bonferroni.to_csv(os.path.join(res_path, f"hirid_sig_med_lab_pairs_bonferroni_{lab_parts}.csv"))
hard.to_csv(os.path.join(res_path, f"hirid_sig_med_lab_pairs_hard_{lab_parts}.csv"))
pval.to_csv(os.path.join(res_path, f"hirid_sig_med_lab_pairs_pval_{lab_parts}.csv"))
