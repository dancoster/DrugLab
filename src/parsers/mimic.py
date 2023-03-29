import matplotlib.pyplot as plt
import pandas as pd
import os

from src.utils.utils import AnalysisUtils, change_col_to_datetime
from src.utils import constants

class MIMICParser(AnalysisUtils):
    
    def __init__(self, data, res, gender="MF", age_b=0, age_a=100, ethnicity="WHITE", lab_mapping=constants.LAB_MAPPING, load="MANUAL_MAPPING_MIMIC", top_k_meds=200):
        """
        Args:
            data (_type_): Raw dataset path
            res (_type_): Preprocessed dataset path
            gender (str, optional): Gender stratification param - Male (M), Female (F), Male Female (MF). Defaults to "MF".
            age_b (int, optional): Starting of age range. Defaults to 0.
            age_a (int, optional): Ending of age range. Defaults to 100.
            ethnicity (str, optional): Ethniity stratification parameter. Defaults to "WHITE".
            lab_mapping (_type_, optional): Lab to ID mapping taken by default from constants.py. Defaults to LAB_MAPPING.
            load (str, optional): load mappings for lab test and medication data - "MANUAL_MAPPING" (labtest and medication names taken from constants.py), "AUTOMATIC_MAPPING" (based on number of subjects associated with the medication/labtest)
            top_k_meds (int, optional): Chooses the "top_k_most" most common drugs (Note only used when "AUTOMATIC_MAPPING" is choosen for load type). Defaults to 200.
        """
        AnalysisUtils.__init__(self, data, res, gender=gender, age_b=age_b, age_a=age_a, ethnicity=ethnicity, lab_mapping=lab_mapping, load=load, top_k_meds=top_k_meds)
        
    def generate_med_data(self):
        """Generate and save preprocessed medication data by reading the raw MIMIC III tables.

        Returns:
            pd.DataFrame: Preprocessed medication data with medicaiton name, time after admission and other info.
        """
        print("Generate med data....")
        # Filtering 1st admission data for each subject/patient
        admits = pd.read_csv(os.path.join(self.data, constants.MIMIC_III_RAW_PATH, "ADMISSIONS.csv.gz")).sort_values(["SUBJECT_ID", "ADMITTIME"]).groupby(["SUBJECT_ID"]).nth(0).reset_index()
        inputevents_mv = pd.read_csv(os.path.join(self.data, constants.MIMIC_III_RAW_PATH, "INPUTEVENTS_MV.csv.gz"))
        patients = pd.read_csv(os.path.join(self.data, constants.MIMIC_III_RAW_PATH, "PATIENTS.csv.gz"))
                
        ### Merge medication and admission data
        inputevents_mv = pd.merge(inputevents_mv, admits, how="inner", on=["HADM_ID", "SUBJECT_ID"])
        inputevents_mv = pd.merge(inputevents_mv, patients, how="inner", on=["SUBJECT_ID"])
        inputevents_mv = change_col_to_datetime(inputevents_mv, 'ADMITTIME')
        inputevents_mv = change_col_to_datetime(inputevents_mv, 'ENDTIME')
        inputevents_mv = change_col_to_datetime(inputevents_mv, 'STARTTIME')
        inputevents_mv = change_col_to_datetime(inputevents_mv, 'DOB')
        inputevents_mv['MedTimeFromAdmit'] = inputevents_mv['ENDTIME']-inputevents_mv['ADMITTIME']

        ### Add medication information from D_ITEMS table in MIMIC III dataset (like label name)
        med_data = pd.read_csv(os.path.join(self.data, constants.MIMIC_III_RAW_PATH, "D_ITEMS.csv.gz"))
        med_data = med_data[med_data["DBSOURCE"].isin(["metavision", "carevue"])]   ## As we are only working with INPUTEVENTS_MV and INPUTEVENTS_CV tables
        
        # Merge medication data with medication labels
        m_p_df = pd.merge(inputevents_mv, med_data, how="inner", on="ITEMID")
        m_p_df = m_p_df.sort_values(by="MedTimeFromAdmit") 
        
        m_p_df['AGE'] = m_p_df.apply(lambda r: round((r['ADMITTIME'].to_pydatetime()-r['DOB'].to_pydatetime()).days/365, 0), axis=1)
        
        m_p_df.to_csv(os.path.join(self.data, constants.MIMIC_III_PREPROCESSED_PATH, constants.MIMIC_III_MED_PREPROCESSED_FILE_PATH))
        print("Generated med data.")
        return m_p_df
    
    def stratify_med_data(self, m_p_df):
        """Apply stratification based on age, gender nad ethnicity

        Args:
            m_p_df (pd.DataFrame): Preprocessed medication data

        Returns:
            pd.DataFrame: Stratified preprocessed medication data.
        """
        
        # Stratification based on age, gender and ethinicity
        m_p_df = m_p_df[m_p_df["AGE"]>=self.age_b]
        m_p_df = m_p_df[m_p_df["AGE"]<=self.age_a]
        m_p_df = m_p_df[m_p_df["GENDER"]==self.gender] if self.gender != "MF" else m_p_df
        m_p_df = m_p_df[m_p_df["ETHNICITY"]==self.ethnicity]
        
        return m_p_df

    def generate_med_k_vect(self, med_preprocessed, k=1):
        """Generate vectorized version of kth medication data
        
        Args:
            med_preprocessed (_type_): Preprocessed medication data with medication names and medication administration times
            k (int, optional): kth Medication administered to patients. Defaults to 1.
        """
        med_k = med_preprocessed.groupby(["HADM_ID", "ITEMID"]).nth(k-1).reset_index()        
        # Making sure we only take 1st admission data of the patient. This also makes sure data related to one admission is only taken
        med_k = med_k.sort_values(by=["ADMITTIME"]).groupby(["SUBJECT_ID", "ITEMID"]).nth(0).reset_index().sort_values(by=["MedTimeFromAdmit"])
        # save
        med_k.to_csv(os.path.join(self.data, constants.MIMIC_III_PREPROCESSED_PATH, f"med{k}_vectorized.csv"))
        # med_k = pd.read_csv(os.path.join(self.data, constants.MIMIC_III_PREPROCESSED_PATH, f"med{k}_vectorized.csv"))
        return med_k
    
    def load_med_k_vect(self, med_preprocessed, k=1, load_from_raw=False):
        """Load vectorized version of kth medication data
        
        Args:
            med_preprocessed (_type_): Preprocessed medication data with medication names and medication administration times
            k (int, optional): kth Medication administered to patients. Defaults to 1.
            load_from_raw (bool, optional): Load preprocessed vectorized med data from raw MIMIC tables instead of using the previous generated files. Defaults to False.
        """
        med_vect_data_path = os.path.join(self.data, constants.MIMIC_III_PREPROCESSED_PATH, F"med{k}_vectorized.csv")
        med_k = pd.read_csv(med_vect_data_path) if not load_from_raw and os.path.exists(med_vect_data_path) else self.generate_med_k_vect(med_preprocessed=med_preprocessed, k=k)
        
        # Choose only first admission data
        h_adm_k = med_k.sort_values(["HADM_ID", "STARTTIME"]).groupby("SUBJECT_ID").nth(0)["HADM_ID"].to_list()
        med_k = med_k[med_k.HADM_ID.isin(h_adm_k)]
        if ("Unnamed: 0" in med_k.columns):
            med_k = med_k.drop(columns=["Unnamed: 0"])
        
        # Grouping medication data based on mapping and setting the MIMIC Extract label. If label is absent in mapping then choose the original name form MIMIC III dataset
        med_k["MIMICExtractLabel"] = med_k.apply(lambda r: self.med_mapping[r["ITEMID"]] if r["ITEMID"] in self.med_mapping else r["LABEL"], axis=1)
        
        # Adding other info
        med_k["STARTTIME"] = pd.to_datetime(med_k["STARTTIME"])
        med_k["ENDTIME"] = pd.to_datetime(med_k["ENDTIME"])
        med_k["ADMITTIME"] = pd.to_datetime(med_k["ADMITTIME"])
        med_k["MedTimeFromAdmit"] = med_k["ENDTIME"]-med_k["ADMITTIME"]
        med_k["hours_in"] = med_k["MedTimeFromAdmit"].dt.total_seconds()/3600
        
        return med_k, h_adm_k
    
    def generate_lab_vect(self, use_partitioned_files=False):
        """
        Generate vectorized version of Lab data
        """
        
        # Read admits information from mimic3 dataset
        admits = pd.read_csv(os.path.join(self.data, constants.MIMIC_III_RAW_PATH, "ADMISSIONS.csv.gz"))
        patients = pd.read_csv(os.path.join(self.data, constants.MIMIC_III_RAW_PATH, "PATIENTS.csv.gz"))

        ### Final Mapping (from the above "lab_itemids" dictionary, ie, output of manual mapping from mimic extract) in the required format
        final_mapping_lab_itemids = {v2:k for k, v in self.lab_mapping.items() for v2 in v}
        final_itemids_list = list(final_mapping_lab_itemids.keys())
        
        ## Labevents
        print("Generate lab data from labevents...")
        if use_partitioned_files:
            labevents = pd.read_csv(os.path.join(self.data, constants.MIMIC_III_PREPROCESSED_PATH, constants.MIMIC_III_LABEVENT_PREPROCESSED))
        else:
            labevents = pd.read_csv(os.path.join(self.data, constants.MIMIC_III_RAW_PATH, "LABEVENTS.csv.gz"))

            # ### Preprocessing labevents data to add requried features like "MIMIC Extract Names" and "age at admit time"
            labevents = labevents[labevents.ITEMID.isin(final_itemids_list)]
            labevents["MIMICExtractName"] = labevents.apply(lambda r: final_mapping_lab_itemids[r["ITEMID"]], axis=1)
            labevents["TABLE"] = labevents.apply(lambda r: "LABEVENTS", axis=1)
            labevents = labevents.rename(columns={"SUBJECT_ID_x":"SUBJECT_ID"})

            #### Age at admit time
            labevents = pd.merge(labevents, admits, how="inner", on=["HADM_ID", "SUBJECT_ID"])
            labevents = pd.merge(labevents, patients, how="inner", on="SUBJECT_ID")
            labevents["DOB"] = pd.to_datetime(labevents["DOB"])
            labevents["ADMITTIME"] = pd.to_datetime(labevents["ADMITTIME"])
            labevents['AGE'] = labevents.apply(lambda r: round((r['ADMITTIME'].to_pydatetime()-r['DOB'].to_pydatetime()).days/365, 0), axis=1)

            columns = constants.LAB_VECT_COLS
            columns.extend(['ROW_ID_x', 'TABLE'])
            labevents[columns].to_csv(os.path.join(self.data, constants.MIMIC_III_PREPROCESSED_PATH, constants.MIMIC_III_LABEVENT_PREPROCESSED))
        print("Generated lab data from labevents.") 
        
        print("Generate lab data from chartevents...")
        ## Chartevents
        ### Reading chartevents data in chunks and saving the output CSV files for each chunck (batch processing). 
        ### The output of each chunk (csv file) is later concatenated and stored.
        if use_partitioned_files:
            res_paths = [os.path.join(self.data, "mimiciii", "1.4","preprocessed", "CHARTEVENTS", f"chartevents_with_mimic_extract_{count}.csv.gz") for count in range(67)]
        else:
            res_paths = []
            count = 0
            with pd.read_csv(os.path.join(self.data, constants.MIMIC_III_RAW_PATH, "CHARTEVENTS.csv.gz"), chunksize=5_000_000) as reader:
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
        
        chartevents = pd.concat([pd.read_csv(path) for path in res_paths], ignore_index=True) ## concatenation of output from each chunk

        ### Preprocessing of chartevents table and adding requried fields like "age at admit time"
        chartevents["TABLE"] = chartevents.apply(lambda r: "CHARTEVENTS", axis=1)
        chartevents = pd.merge(chartevents, admits, how="inner", on="HADM_ID")
        chartevents = chartevents.rename(columns={"SUBJECT_ID_x":"SUBJECT_ID"})

        #### Age at admit time
        chartevents = pd.merge(chartevents, patients, how="inner", on="SUBJECT_ID")
        chartevents["DOB"] = pd.to_datetime(chartevents["DOB"])
        chartevents["ADMITTIME"] = pd.to_datetime(chartevents["ADMITTIME"])
        chartevents['AGE'] = chartevents.apply(lambda r: round((r['ADMITTIME'].to_pydatetime()-r['DOB'].to_pydatetime()).days/365, 0), axis=1)
        print("Generated lab data from chartevents.")
        
        # Select cols in chartevents
        columns = constants.LAB_VECT_COLS
        columns.extend(['ROW_ID_x', 'TABLE'])
        chartevents[columns].to_csv(os.path.join(self.data, constants.MIMIC_III_PREPROCESSED_PATH, "chartevents_with_mimic_extract_lab.csv.gz"))

        ## Merge output from chartevents and labevents
        ### select required columns from CHARTEVENTS (result -> t_r) and LABEVETS (labevents -> t_l)
        t_chart = chartevents[constants.LAB_VECT_COLS]
        t_lab = labevents[constants.LAB_VECT_COLS]
        merged_chart_lab_events = pd.concat([t_lab, t_chart]).drop_duplicates(keep="first") # Concatenate labevents and chartevents data. Choose only the first if there are duplicates
        del t_lab, t_chart
        
        ## Sanity check on calculated age (removing inhumane age values)
        merged_chart_lab_events = merged_chart_lab_events[merged_chart_lab_events["AGE"]<100]
        merged_chart_lab_events = merged_chart_lab_events[merged_chart_lab_events["AGE"]>0]
        merged_chart_lab_events = merged_chart_lab_events.groupby(["HADM_ID", "ADMITTIME", "CHARTTIME", "VALUENUM", "MIMICExtractName"]).nth(0).reset_index()
        merged_chart_lab_events = merged_chart_lab_events.dropna(subset=['VALUENUM'])
        merged_chart_lab_events.to_csv(os.path.join(self.data, constants.MIMIC_III_PREPROCESSED_PATH, constants.MIMIC_III_PREPROCESSED_LABDATA))

        return merged_chart_lab_events

    def load_lab(self, h_med_adm1, load_from_raw=True, load_raw_chartevents=False):
        """
        Load lab test data from LABEVENTS and CHARTEVENTS tables
        """
        
        if not load_from_raw and os.path.exists(os.path.join(self.data, constants.MIMIC_III_PREPROCESSED_PATH, constants.MIMIC_III_PREPROCESSED_LABDATA)):
            labs = pd.read_csv(os.path.join(self.data, constants.MIMIC_III_PREPROCESSED_PATH, constants.MIMIC_III_PREPROCESSED_LABDATA)) 
            labs = labs.drop(columns=["Unnamed: 0"])
        else:
            bool_val = not load_raw_chartevents
            labs = self.generate_lab_vect(use_partitioned_files=bool_val)
        
        labs = labs[labs.HADM_ID.isin(h_med_adm1)]
        
        # Stratification
        labs = labs[labs["AGE"]>=self.age_b]
        labs = labs[labs["AGE"]<=self.age_a]
        labs = labs[labs["GENDER"]==self.gender] if self.gender != "MF" else labs
        labs = labs[labs["ETHNICITY"]==self.ethnicity]
        
        # Adding other column variables
        labs["CHARTTIME"] = pd.to_datetime(labs["CHARTTIME"])
        labs["ADMITTIME"] = pd.to_datetime(labs["ADMITTIME"])
        labs["LabTimeFromAdmit"] = labs["CHARTTIME"]-labs["ADMITTIME"]
        labs["hours_in"] = labs["LabTimeFromAdmit"].dt.total_seconds()/3600
        
        return labs

    def parse(self, use_pairs=False, load_from_raw=True, load_raw_chartevents=False):
        """Loading medication and lab test. Performing basic preprocessing on data.
        
        Args:
            use_pairs (bool, optional): _description_. Defaults to True.
            load_from_raw (bool, optional): Load preprocessed med files from raw MIMIC tables instead of using the previosu prepocessed files. Defaults to True.

        Returns:
            pd.DataFrame: 1st medication data
            pd.DataFrame: 2nd medication data
            pd.DataFrame: Lab tests data
        """
        print(f"Loading med data...")
        med_preprocessed_path = os.path.join(self.data, constants.MIMIC_III_PREPROCESSED_PATH, constants.MIMIC_III_MED_PREPROCESSED_FILE_PATH)
        med_preprocessed = pd.read_csv(med_preprocessed_path) if not load_from_raw and os.path.exists(med_preprocessed_path) else self.generate_med_data()
        med_preprocessed = self.stratify_med_data(med_preprocessed)
        print(f"Loaded med data.")
        
        print(f"Load 1st and 2nd medication data...")
        med1, hadm1 = self.load_med_k_vect(med_preprocessed=med_preprocessed, k=1, load_from_raw=load_from_raw)
        med2, _ = self.load_med_k_vect(med_preprocessed=med_preprocessed, k=2, load_from_raw=load_from_raw)
        print(f"Loaded 1st and 2nd medication data.")
        print(f"Load Lab data...")
        labs = self.load_lab(hadm1, load_from_raw=load_from_raw, load_raw_chartevents=load_raw_chartevents)
        print(f"Loaded Lab data.")
        
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
