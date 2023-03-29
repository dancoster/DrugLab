from src.parsers import mimic, hirid
from src.modeling import discovery, plots, querier
from src.utils import constants
import sys
# import win32com.client
import os
import pandas as pd

def setup_io_config(root_path):
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
        mimic_data = os.path.join(f"{root_path}", "data") 
    mimic_path = os.path.join(f"{root_path}", "results")

    # HIRID
    hirid_data = f'{root_path}/data/hirid-a-high-time-resolution-icu-dataset-1.1.1/raw_stage/'
    hirid_path = f'{root_path}/data/hirid-a-high-time-resolution-icu-dataset-1.1.1'
    
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
data, res, raw_path, res_path = setup_io_config(root_path=root_path)

# Stratification Config
gender, age_a, age_b, ethnicity, lab_mapping, before_windows, after_windows = setup_stratification_config()
# MIMIC
mimic_parser = mimic.MIMICParser(data=data, res=res, gender=gender, age_b=age_b, age_a=age_a, ethnicity=ethnicity, load="AUTOMATIC_MAPPING_MIMIC")
m_med1, m_med2, m_labs = mimic_parser.parse(use_pairs=False, load_from_raw=True, load_raw_chartevents=True)

## Querier
mimic_data_querier = querier.DatasetQuerier(
    data = data,
    res = res,
    t_labs=m_labs, 
    t_med1=m_med1, 
    t_med2=m_med2,
    gender=gender, 
    age_b=age_b, 
    age_a=age_a, 
    ethnicity=ethnicity, 
    lab_mapping=lab_mapping
)
# query pairs for all medication and lab tests
m_final_lab_med_data = mimic_data_querier.generate_med_lab_data(before_windows, after_windows)

## Discovery Analysis for the queried medication and lab test pairs in the chosen before and after windows
analyzer = discovery.ClinicalDiscoveryAnalysis(m_final_lab_med_data)
types_l = ["abs", "mean", "std", "trends"]
pvals_med_lab = analyzer.analyze(before_windows, after_windows, min_patients=100, types_l=types_l)
pval, hard, bonferroni, fdr = analyzer.generate_significant(pvals_med_lab.dropna(subset=["TTest Paired"]))

fdr.to_csv(f"temp_mimic_sig_med_lab_pairs_fdr_{1}.csv")
bonferroni.to_csv(f"temp_mimic_sig_med_lab_pairs_bonferroni_{1}.csv")
hard.to_csv(f"temp_mimic_sig_med_lab_pairs_hard_{1}.csv")
pval.to_csv(f"temp_mimic_sig_med_lab_pairs_pval_{1}.csv")