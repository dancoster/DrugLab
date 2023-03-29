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
data_path = "/home/gaga/data/physionet/hirid"
data, res, raw_path, res_path = setup_io_config(root_path=root_path, data_path=data_path)

# Stratification Config
gender, age_a, age_b, ethnicity, lab_mapping, before_windows, after_windows = setup_stratification_config()

# HIRID
hirid_mapping = constants.HIRID_MAPPING
lab_parts = (130,180)
hirid_parser = hirid.HiRiDParser(data=raw_path, res=res_path, gender=gender, age_b=age_b, age_a=age_a, load="MANUAL_MAPPING_HIRID")
h_med1, h_med2, h_labs = hirid_parser.parse(lab_parts=lab_parts)
hirid_data_querier = querier.DatasetQuerier(
    data = raw_path,
    res = res_path,
    t_labs=h_labs, 
    t_med1=h_med1, 
    t_med2=h_med2,
    gender=gender, 
    age_b=age_b, 
    age_a=age_a, 
    ethnicity=ethnicity, 
)
final_h_final_lab_med_data, raw_h_final_lab_med_data = hirid_data_querier.generate_med_lab_data(before_windows, after_windows, lab_parts=lab_parts)
## Discovery Analysis for the queried medication and lab test pairs in the chosen before and after windows
analyzer = discovery.ClinicalDiscoveryAnalysis(final_h_final_lab_med_data)
types_l = ["abs", "mean", "std", "trends"]
pvals_med_lab = analyzer.analyze(before_windows, after_windows, min_patients=100, types_l=types_l)

pval, hard, bonferroni, fdr = analyzer.generate_significant(pvals_med_lab.dropna(subset=["TTest Paired"]))
fdr.to_csv(f"temp_hirid_sig_med_lab_pairs_fdr_{lab_parts}.csv")
bonferroni.to_csv(f"temp_hirid_sig_med_lab_pairs_bonferroni_{lab_parts}.csv")
hard.to_csv(f"temp_hirid_sig_med_lab_pairs_hard_{lab_parts}.csv")
pval.to_csv(f"temp_hirid_sig_med_lab_pairs_pval_{lab_parts}.csv")