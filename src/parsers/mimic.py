import matplotlib.pyplot as plt
import pandas as pd
import os

from src.utils.utils import AnalysisUtils, change_col_to_datetime
from src.utils.constants import LAB_MAPPING, LAB_VECT_COLS


class MIMICParser(AnalysisUtils):
    def __init__(self, data, res, gender="MF", age_b=0, age_a=100, ethnicity="WHITE", lab_mapping=None, load=True):
        self.m_p_df = None
        AnalysisUtils.__init__(self, data, res, gender=gender, age_b=age_b, age_a=age_a, ethnicity=ethnicity, lab_mapping=LAB_MAPPING, load=load)
        
    def generate_med_data(self):
        admits = pd.read_csv(os.path.join(self.data, "mimiciii/1.4/raw", "ADMISSIONS.csv.gz"))
        inputevents_mv = pd.read_csv(os.path.join(self.data, "mimiciii/1.4/raw", "INPUTEVENTS_MV.csv.gz"))
                
        ### Merge medication and admission data
        inputevents_mv = pd.merge(inputevents_mv, admits, how="inner", on=["HADM_ID", "SUBJECT_ID"])
        inputevents_mv = change_col_to_datetime(inputevents_mv, 'ADMITTIME')
        inputevents_mv = change_col_to_datetime(inputevents_mv, 'ENDTIME')
        inputevents_mv = change_col_to_datetime(inputevents_mv, 'STARTTIME')
        inputevents_mv['MedTimeFromAdmit'] = inputevents_mv['ENDTIME']-inputevents_mv['ADMITTIME']

        ### Add medication information from D_ITEMS table in MIMIC III dataset (like label name)
        med_data = pd.read_csv(os.path.join(self.data, "mimiciii/1.4/raw", "D_ITEMS.csv.gz"))
        med_data = med_data[med_data["DBSOURCE"]=="metavision"]
        
        # Merge medication data with medication labels
        m_p_df = pd.merge(inputevents_mv, med_data, how="inner", on="ITEMID")
        self.m_p_df = m_p_df.sort_values(by="MedTimeFromAdmit")
       
    def generate_med1_vect(self):
        """
        Generate vectorized version of Medication data
        """
        
        if self.m_p_df is None:
            self.generate_med_data()
        
        ## Filter Meds
        ### Med 1
        med1 = self.m_p_df.groupby(["HADM_ID", "ITEMID"]).nth(0).reset_index()
        med1 = med1.sort_values("MedTimeFromAdmit")
        med1 = med1[med1["MedTimeFromAdmit"].dt.total_seconds()>0]
        med1 = med1.sort_values(by=["ADMITTIME"]).groupby(["SUBJECT_ID", "ITEMID"]).nth(0).reset_index().sort_values(by=["MedTimeFromAdmit"])
        med1.to_csv(os.path.join(self.data, "mimiciii/1.4/preprocessed", "med1_vectorized.csv"))
        return med1
    
    def generate_med2_vect(self):
        """
        Generate vectorized version of Medication data
        """
        
        if self.m_p_df is None:
            self.generate_med_data()
        
        ## Filter Meds
        ### Med 2
        med2 = self.m_p_df.groupby(["HADM_ID", "ITEMID"]).nth(1).reset_index()
        med2 = med2.sort_values("MedTimeFromAdmit")
        med2 = med2[med2["MedTimeFromAdmit"].dt.total_seconds()>0]
        med2 = med2.sort_values(by=["ADMITTIME"]).groupby(["SUBJECT_ID", "ITEMID"]).nth(0).reset_index().sort_values(by=["MedTimeFromAdmit"])
        med2.to_csv(os.path.join(self.data, "mimiciii/1.4/preprocessed", "med2_vectorized.csv"))
        
        return med2

    def load_med1(self, use_med_vect=True):
        """
        Load 1st Medication data
        """
        med1 = pd.read_csv(os.path.join(self.data, "mimiciii/1.4/preprocessed", "med1_vectorized.csv")) if use_med_vect and os.path.exists(os.path.join(self.data, "mimiciii/1.4/preprocessed", "med1_vectorized.csv")) else self.generate_med1_vect()
        h_adm_1 = med1.sort_values(["HADM_ID", "STARTTIME"]).groupby("SUBJECT_ID").nth(0)["HADM_ID"].to_list()
        med1 = med1[med1.HADM_ID.isin(h_adm_1)]
        if ("Unnamed: 0" in med1.columns):
            med1 = med1.drop(columns=["Unnamed: 0"])
        med1 = med1[med1["AGE"]>=self.age_b]
        med1 = med1[med1["AGE"]<=self.age_a]
        med1 = med1[med1["GENDER"]==self.gender] if self.gender != "MF" else med1
        med1 = med1[med1["ETHNICITY"]==self.ethnicity]
        med1["MIMICExtractLabel"] = med1.apply(lambda r: self.res_dict_mapping_med[r["ITEMID"]] if r["ITEMID"] in self.res_dict_mapping_med else r["LABEL"], axis=1)
        med1["STARTTIME"] = pd.to_datetime(med1["STARTTIME"])
        med1["ENDTIME"] = pd.to_datetime(med1["ENDTIME"])
        med1["ADMITTIME"] = pd.to_datetime(med1["ADMITTIME"])
        med1["MedTimeFromAdmit"] = med1["ENDTIME"]-med1["ADMITTIME"]
        med1["hours_in"] = med1["MedTimeFromAdmit"].dt.total_seconds()/3600
        return med1, h_adm_1
    
    def load_med2(self, use_med_vect=True):
        """
        Load 2nd Medication data
        """        
        med2 = pd.read_csv(os.path.join(self.data, "mimiciii/1.4/preprocessed", "med2_vectorized.csv")) if use_med_vect and os.path.exists(os.path.join(self.data, "mimiciii/1.4/preprocessed", "med2_vectorized.csv")) else self.generate_med2_vect()
        h_adm_2 = med2.sort_values(["SUBJECT_ID", "STARTTIME"]).groupby("SUBJECT_ID").nth(0)["HADM_ID"].to_list()
        med2 = med2.drop(columns=["Unnamed: 0"])
        med2 = med2[med2["AGE"]>=self.age_b]
        med2 = med2[med2["AGE"]<=self.age_a]
        med2 = med2[med2["GENDER"]==self.gender] if self.gender != "MF" else med2
        med2 = med2[med2["ETHNICITY"]==self.ethnicity]
        med2["MIMICExtractLabel"] = med2.apply(lambda r: self.res_dict_mapping_med[r["ITEMID"]] if r["ITEMID"] in self.res_dict_mapping_med else r["LABEL"], axis=1)
        med2["STARTTIME"] = pd.to_datetime(med2["STARTTIME"])
        med2["ENDTIME"] = pd.to_datetime(med2["ENDTIME"])
        med2["ADMITTIME"] = pd.to_datetime(med2["ADMITTIME"])
        med2["MedTimeFromAdmit"] = med2["ENDTIME"]-med2["ADMITTIME"]
        return med2, h_adm_2
    
    def generate_lab_vect(self):
        """
        Generate vectorized version of Lab data
        """
        
        # Read admits information from mimic3 dataset
        admits = pd.read_csv(os.path.join(self.data, "mimiciii/1.4/raw", "ADMISSIONS.csv.gz"))
        patients = pd.read_csv(os.path.join(self.data, "mimiciii/1.4", "raw/PATIENTS.csv.gz"))

        ### Final Mapping (from the above "lab_itemids" dictionary, ie, output of manual mapping from mimic extract) in the required format
        final_mapping_lab_itemids = {v2:k1 for k, v in LAB_MAPPING.items() for k1, v1 in v.items() for v2 in v1}
        final_itemids_list = list(final_mapping_lab_itemids.keys())

        ## Labevents
        labevents = pd.read_csv(os.path.join(self.data, "mimiciii/1.4/preprocessed", "lab_patient_data.csv"))

        ### Preprocessing labevents data to add requried features like "MIMIC Extract Names" and "age at admit time"
        labevents = labevents[labevents.ITEMID.isin(final_itemids_list)]
        labevents["MIMICExtractName"] = labevents.apply(lambda r: final_mapping_lab_itemids[r["ITEMID"]], axis=1)
        labevents["TABLE"] = labevents.apply(lambda r: "LABEVENTS", axis=1)
        labevents = labevents.rename(columns={"SUBJECT_ID_x":"SUBJECT_ID"})

        #### Age at admit time
        labevents = pd.merge(labevents, patients, how="inner", on="SUBJECT_ID")
        labevents["DOB"] = pd.to_datetime(labevents["DOB"])
        labevents["ADMITTIME"] = pd.to_datetime(labevents["ADMITTIME"])
        labevents['AGE'] = labevents.apply(lambda r: round((r['ADMITTIME'].to_pydatetime()-r['DOB'].to_pydatetime()).days/365, 0), axis=1)

        columns = LAB_VECT_COLS
        columns.extend(['ROW_ID_x', 'TABLE'])
        labevents[columns].to_csv(os.path.join(self.data, "mimiciii/1.4/preprocessed", "lab_patient_data_with_mimic_extract_names.csv"))
        # labevents = pd.read_csv(os.path.join(self.data, "mimiciii/1.4/preprocessed", "lab_patient_data_with_mimic_extract_names.csv"))

        ## Chartevents
        ### Reading chartevents data in chunks and saving the output CSV files for each chunck (batch processing). 
        ### The output of each chunk (csv file) is later concatenated and stored.
        res_paths = []
        count = 0
        with pd.read_csv(os.path.join(self.data, "mimiciii/1.4/raw", "CHARTEVENTS.csv.gz"), chunksize=5_000_000) as reader:
            for chunk in reader:
                print(count, chunk.shape)
                path = os.path.join(self.data, "mimiciii", "1.4","preprocessed", "CHARTEVENTS", f"chartevents_with_mimic_extract_{count}.csv.gz")
                chunk = chunk[chunk.ITEMID.isin(final_itemids_list)]
                t = chunk.apply(lambda r: final_mapping_lab_itemids[r["ITEMID"]], axis=1)
                if t.shape[0]>0:
                    chunk["MIMICExtractName"] = t
                    chunk.to_csv(path)
                    print(count, chunk.shape)
                    res_paths.append(path)
                else:
                    print(count, 0)    
                count += 1

        result = pd.concat([pd.read_csv(f) for f in res_paths], ignore_index=True) ## concatenation of output from each chunk

        ### Preprocessing of chartevents table and adding requried fields like "age at admit time"
        result["TABLE"] = result.apply(lambda r: "CHARTEVENTS", axis=1)
        result = pd.merge(result, admits, how="inner", on="HADM_ID")
        result = result.rename(columns={"SUBJECT_ID_x":"SUBJECT_ID"})

        #### Age at admit time
        result = pd.merge(result, patients, how="inner", on="SUBJECT_ID")
        result["DOB"] = pd.to_datetime(result["DOB"])
        result["ADMITTIME"] = pd.to_datetime(result["ADMITTIME"])
        result['AGE'] = result.apply(lambda r: round((r['ADMITTIME'].to_pydatetime()-r['DOB'].to_pydatetime()).days/365, 0), axis=1)

        columns = LAB_VECT_COLS
        columns.extend(['ROW_ID_x', 'SUBJECT_ID_y', 'TABLE'])
        result[columns].to_csv(os.path.join(self.data, "mimiciii", "1.4","preprocessed", "CHARTEVENTS", "chartevents_with_mimic_extract_lab.csv.gz"))


        ## Merge output from chartevents and labevents

        ### select required columns from CHARTEVENTS (result -> t_r) and LABEVETS (labevents -> t_l)
        t_r = result[LAB_VECT_COLS]
        t_l = labevents[LAB_VECT_COLS]
        t_c_l = pd.concat([t_l, t_r]).drop_duplicates(keep="first") # Concatenate labevents and chartevents data. Choose only the first if there are duplicates
        del t_l, t_r
        ## Sanity check on calculated age (removing inhumane age values)
        t_c_l = t_c_l[t_c_l["AGE"]<100]
        t_c_l = t_c_l[t_c_l["AGE"]>0]
        t_c_l = t_c_l.groupby(["HADM_ID", "ADMITTIME", "CHARTTIME", "VALUENUM", "MIMICExtractName"]).nth(0).reset_index()
        t_c_l = t_c_l.dropna(subset=['VALUENUM'])
        t_c_l.to_csv(os.path.join(self.data, "mimiciii/1.4/preprocessed", "lab_patient_data_mimic_extract_2.csv"))

        return t_c_l

    def load_lab(self, h_med_adm1, h_med_adm2, use_lab_vect=True):
        """
        Load lab test data from LABEVENTS and CHARTEVENTS tables
        """
        if use_lab_vect and os.path.exists(os.path.join(self.data, "mimiciii/1.4/preprocessed", "lab_patient_data_mimic_extract_2.csv")):
            labs = pd.read_csv(os.path.join(self.data, "mimiciii/1.4/preprocessed", "lab_patient_data_mimic_extract_2.csv")) 
        else:
            self.generate_lab_vect()
        labs = labs[labs["AGE"]<100]
        labs = labs.drop(columns=["Unnamed: 0"])
        labs = labs[labs.HADM_ID.isin(h_med_adm1+h_med_adm2)]
        labs = labs[labs["AGE"]>=self.age_b]
        labs = labs[labs["AGE"]<=self.age_a]
        labs = labs[labs["GENDER"]==self.gender] if self.gender != "MF" else labs
        labs = labs[labs["ETHNICITY"]==self.ethnicity]
        labs["CHARTTIME"] = pd.to_datetime(labs["CHARTTIME"])
        labs["ADMITTIME"] = pd.to_datetime(labs["ADMITTIME"])
        labs["LabTimeFromAdmit"] = labs["CHARTTIME"]-labs["ADMITTIME"]
        labs["hours_in"] = labs["LabTimeFromAdmit"].dt.total_seconds()/3600
        return labs

    def parse(self, use_pairs=True):
        """
        Loading medication and lab test. Performing basic preprocessing on data.
        """
        med1, hadm1 = self.load_med1()
        med2, hadm2 = self.load_med2()
        labs = self.load_lab(hadm1, hadm2)
        
        t_med1, t_med2, t_labs = med1.copy(), med2.copy(), labs.copy()
        if use_pairs:
            med_vals_new, labtest_vals_new = self.generate_med_lab_pairs()
            t_med1 = med1[med1["MIMICExtractLabel"].isin(med_vals_new)]
            t_med2 = med2[med2["MIMICExtractLabel"].isin(med_vals_new)]
            t_labs = labs[labs["MIMICExtractName"].isin(labtest_vals_new)]
            
        t_med1 = t_med1.rename(columns={"LABEL":"OldLabel", "ITEMID":"OldITEMID", "MIMICExtractLabel":"ITEMID"})
        t_med2 = t_med2.rename(columns={"LABEL":"OldLabel", "ITEMID":"OldITEMID", "MIMICExtractLabel":"ITEMID"})
        t_labs = t_labs.rename(columns={"LABEL":"OldLabel", "ITEMID":"OldITEMID", "MIMICExtractName":"ITEMID"})

        return t_med1, t_med2, t_labs
