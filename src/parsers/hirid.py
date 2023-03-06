import pandas as pd
import os

from src.utils.utils import AnalysisUtils
from src.utils.constants import HIRID_LAB_IDS

class HiRiDParser(AnalysisUtils):
    def __init__(self, data, res, gender="MF", age_b=0, age_a=100, load="MANUAL_MAPPING_HIRID"):
        AnalysisUtils.__init__(self, data=data, res=res, gender=gender, age_b=age_b, age_a=age_a, load=load, )
        self.load_util_datasets()

    def load_util_datasets(self):
        path1 = self.res
        self.g_table = pd.read_csv(os.path.join(path1, 'general_table.csv'))
        h_var_ref = pd.read_csv(os.path.join(path1, 'hirid_variable_reference.csv'))
        self.h_var_ref = h_var_ref.rename(columns={"ID":"variableid"})

        self.h_var_ref_pre = pd.read_csv(os.path.join(path1, 'hirid_variable_reference_preprocessed.csv'))
        self.o_var_ref = pd.read_csv(os.path.join(path1, 'ordinal_vars_ref.csv'))

    def load_med(self):

        pharma_records_paths = [i for iq, i in enumerate(os.walk(os.path.join(self.data, "pharma_records"))) if iq==1][0][2]
        pharma_records = pd.concat([pd.read_csv(os.path.join(self.data, "pharma_records", 'csv', file)) for file in pharma_records_paths])
        pharma_records = pharma_records.rename(columns={"pharmaid":"variableid"})

        pharma_records_with_name = pd.merge(pharma_records, self.h_var_ref, on="variableid", how="inner")
        pharma_records_with_name = pd.merge(pharma_records_with_name, self.g_table, on="patientid", how="inner")
        pharma_records_with_name.givenat = pd.to_datetime(pharma_records_with_name.givenat)
        self.pharma_records_with_name = pharma_records_with_name.rename(columns={
            "givenat":"STARTTIME",
            "admissiontime":"ADMITTIME",
            "enteredentryat":"ENDTIME",
            "variableid":"ITEMID",
            "patientid":"HADM_ID",
            "Variable Name":"LABEL",
            "age":"AGE",
            "sex":"GENDER",
        })

    def load_med1(self):
        """
        Load 1st Medication data
        """

        med1 = self.pharma_records_with_name.sort_values(["HADM_ID", "STARTTIME"]).groupby(["HADM_ID", "ITEMID"]).nth(0).reset_index()

        # stratification
        h_adm_1 = med1["HADM_ID"].to_list()
        med1 = med1[med1["AGE"]>=self.age_b]
        med1 = med1[med1["AGE"]<=self.age_a]
        med1 = med1[med1["GENDER"]==self.gender] if self.gender != "MF" else med1

        med1["STARTTIME"] = pd.to_datetime(med1["STARTTIME"])
        med1["ENDTIME"] = pd.to_datetime(med1["ENDTIME"])
        med1["ADMITTIME"] = pd.to_datetime(med1["ADMITTIME"])
        med1["MedTimeFromAdmit"] = med1["STARTTIME"]-med1["ADMITTIME"]
        med1["hours_in"] = med1["MedTimeFromAdmit"].dt.total_seconds()/3600
        self.med1 = med1

        return med1, h_adm_1
    
    def load_med2(self):
        """
        Load 2nd Medication data
        """
        med2 = self.pharma_records_with_name.sort_values(["HADM_ID", "STARTTIME"]).groupby(["HADM_ID", "ITEMID"]).nth(1).reset_index()

        # stratification
        h_adm_2 = med2["HADM_ID"].to_list()
        med2 = med2[med2["AGE"]>=self.age_b]
        med2 = med2[med2["AGE"]<=self.age_a]
        med2 = med2[med2["GENDER"]==self.gender] if self.gender != "MF" else med2

        med2["STARTTIME"] = pd.to_datetime(med2["STARTTIME"])
        med2["ENDTIME"] = pd.to_datetime(med2["ENDTIME"])
        med2["ADMITTIME"] = pd.to_datetime(med2["ADMITTIME"])
        med2["MedTimeFromAdmit"] = med2["STARTTIME"]-med2["ADMITTIME"]
        med2["hours_in"] = med2["MedTimeFromAdmit"].dt.total_seconds()/3600
        self.med2 = med2

        return med2, h_adm_2
    
    def read_lab(self, path, adm):
        labs = pd.read_csv(path)
        labs = labs[labs.patientid.isin(adm)]
        labs = labs[labs.variableid.isin(self.lab_mapping)]
        return labs

    def load_lab(self, h_med_adm1, h_med_adm2, n_parts=50):
        """
        Load lab test data from LABEVENTS and CHARTEVENTS tables
        """
        observation_tables_paths = sorted([i for iq, i in enumerate(os.walk(os.path.join(self.data, "observation_tables 2"))) if iq==1][0][2])
        observation_tables_part = pd.concat([self.read_lab(os.path.join(self.data, "observation_tables 2", 'csv', file), h_med_adm1+h_med_adm2) for file in observation_tables_paths[:n_parts]])

        observation_tables_part_with_name = pd.merge(observation_tables_part, self.h_var_ref, on="variableid", how="inner")
        observation_tables_part_with_name = pd.merge(observation_tables_part_with_name, self.g_table, on="patientid", how="inner")
        observation_tables_part_with_name.datetime = pd.to_datetime(observation_tables_part_with_name.datetime)
        
        observation_tables_part_with_name["Variable Name"].value_counts()
        observation_tables_part_with_name = observation_tables_part_with_name.rename(columns={
            "datetime":"CHARTTIME",
            "admissiontime":"ADMITTIME",
            "variableid":"ITEMID",
            "patientid":"HADM_ID",
            "Variable Name":"LABEL",
            "value":"VALUENUM",
            "Unit":"VALUEUOM",
            "age":"AGE",
            "sex":"GENDER"
        })
        labs = observation_tables_part_with_name.copy()

        labs = labs[labs["AGE"]>=self.age_b]
        labs = labs[labs["AGE"]<=self.age_a]
        labs = labs[labs["GENDER"]==self.gender] if self.gender != "MF" else labs
        
        labs["CHARTTIME"] = pd.to_datetime(labs["CHARTTIME"])
        labs["ADMITTIME"] = pd.to_datetime(labs["ADMITTIME"])
        labs["LabTimeFromAdmit"] = labs["CHARTTIME"]-labs["ADMITTIME"]
        labs["hours_in"] = labs["LabTimeFromAdmit"].dt.total_seconds()/3600
        
        return labs

    def parse(self, use_pairs=False):
        """
        Loading medication and lab test. Performing basic preprocessing on data.
        """
        self.load_med()
        med1, hadm1 = self.load_med1()
        med2, hadm2 = self.load_med2()
        labs = self.load_lab(hadm1, hadm2)
        
        t_med1, t_med2, t_labs = med1.copy(), med2.copy(), labs.copy()
        
        if use_pairs:
            med_vals_new, labtest_vals_new = self.generate_med_lab_pairs()
            t_med1 = med1[med1["LABEL"].isin(med_vals_new)]
            t_med2 = med2[med2["LABEL"].isin(med_vals_new)]
            t_labs = labs[labs["LABEL"].isin(labtest_vals_new)]
            
        t_med1 = t_med1.rename(columns={"ITEMID":"OldITEMID", "LABEL":"ITEMID"})
        t_med2 = t_med2.rename(columns={"ITEMID":"OldITEMID", "LABEL":"ITEMID"})
        t_labs = t_labs.rename(columns={"ITEMID":"OldITEMID", "LABEL":"ITEMID"})

        return t_med1, t_med2, t_labs