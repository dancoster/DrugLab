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

# HIRID
hirid_mapping = constants.HIRID_MAPPING
lab_parts = (0,100)
hirid_parser = hirid.HiRiDParser(data=raw_path, res=res_path, gender=gender, age_b=age_b, age_a=age_a, load="MANUAL_MAPPING_HIRID")
h_med1, h_med2, h_labs = hirid_parser.parse(lab_parts=lab_parts)

lab_with_index = h_labs[["HADM_ID", "AGE", "GENDER", "LabTimeFromAdmit", "VALUENUM", "ITEMID"]].pivot_table("VALUENUM", ["HADM_ID", "AGE", "GENDER",  "LabTimeFromAdmit"], "ITEMID")
lab_with_index.to_csv(os.path.join(res_path, f"hirid_extract_{lab_parts}.csv"))