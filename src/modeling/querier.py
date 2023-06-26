import pandas as pd
import os
from tqdm import tqdm
import json
from src.utils.utils import AnalysisUtils, get_normalized_trend
from src.utils import constants
from tqdm.auto import tqdm
tqdm.pandas()

class DatasetQuerier(AnalysisUtils):

    def __init__(self, data, res, t_labs, meds, gender="MF", age_b=0, age_a=100, ethnicity="WHITE", lab_mapping=None):
        self.final_pairs_data, self.interim_pairs_data = [], []
        self.t_labs = t_labs
        self.meds = meds
        super().__init__(data, res, gender=gender, age_b=age_b, age_a=age_a, ethnicity=ethnicity, lab_mapping=lab_mapping)
    
    def check_medk(self, t_medk, row):
        """
        Check if a kth medication was administered to patients
        """
        if t_medk is None:
            return False
        if row["HADM_ID"] in t_medk["HADM_ID"].to_list():
            if row["ITEMID"] in t_medk[t_medk["HADM_ID"]==row["HADM_ID"]]["ITEMID"].to_list():
                return True
        return False

    def get_medk(self, t_medk, row):
        '''
        Return 2nd medication data
        '''
        temp = t_medk[t_medk["HADM_ID"]==row["HADM_ID"]] 
        return temp[temp["ITEMID"]==row["ITEMID"]].iloc[0]
    
    def get_vals(self, r, t_labs, t_med0, t_med2, before_windows, after_windows):
        """
        Calculate the lab test values in time windows before and after medication administration. Return a dataframe with labtest values of before and after windows as a dict
        Params: 
        - before_windows: list of tuples (each tuple is a window)
        - after_windows: list of tuples (each tuple is a window)
        """

        row = r.copy()
        for b_w in before_windows:
            lab_vals = t_labs[t_labs["HADM_ID"]==row["HADM_ID"]]
            med0_bool = self.check_medk(t_med0, row)
            lab_vals = lab_vals[lab_vals["LabTimeFromAdmit"].dt.total_seconds()<row["MedTimeFromAdmit"].total_seconds()]

            b_window_start = row["MedTimeFromAdmit"].total_seconds() - (b_w[0]*3600)
            b_window_end = row["MedTimeFromAdmit"].total_seconds() - (b_w[1])*3600
            lab_vals = lab_vals[lab_vals["LabTimeFromAdmit"].dt.total_seconds()<b_window_start]
            lab_vals = lab_vals[lab_vals["LabTimeFromAdmit"].dt.total_seconds()>b_window_end]
            lab_vals["hours_from_med"] = (row["STARTTIME"]-lab_vals["CHARTTIME"]).dt.total_seconds()/3600
            lab_vals = lab_vals.sort_values(["ITEMID", "hours_from_med"])
                        
            if med0_bool:
                med0_val = self.get_medk(t_med0, row)
                lab_vals = lab_vals[lab_vals["LabTimeFromAdmit"].dt.total_seconds()>med0_val["MedTimeFromAdmit"].total_seconds()]

            t = lab_vals.groupby(["ITEMID"]).count()[["HADM_ID"]]
            val_counts_m = t[t["HADM_ID"]>=1]
            if val_counts_m.shape[0]==0:
                row[f"before_abs_{b_w}"] = {}
                row[f"before_mean_{b_w}"] = {}
                row[f"before_std_{b_w}"] = {}
                row[f"before_trends_{b_w}"] = {}
                row[f"before_time_{b_w}"] = {}
            else:
                l_m = lab_vals[lab_vals.ITEMID.isin(val_counts_m.index)]
                row[f"before_abs_{b_w}"] = l_m.groupby(["ITEMID"])[["VALUENUM", "hours_from_med"]].first()["VALUENUM"].dropna().to_dict()
                row[f"before_mean_{b_w}"] = l_m.groupby(["ITEMID"])[["VALUENUM"]].mean()["VALUENUM"].dropna().to_dict()
                row[f"before_std_{b_w}"] = l_m.groupby(["ITEMID"])[["VALUENUM"]].std()["VALUENUM"].dropna().to_dict()
                row[f"before_trends_{b_w}"] = l_m[["VALUENUM", "hours_from_med", "ITEMID"]].dropna().groupby(["ITEMID"])[["VALUENUM", "hours_from_med"]].apply(lambda r : get_normalized_trend(r)).dropna().to_dict()
                row[f"before_time_{b_w}"] = l_m.groupby(["ITEMID"])[["VALUENUM", "hours_from_med"]].first()["hours_from_med"].dropna().to_dict()

        for a_w in after_windows:

            lab_vals = t_labs[t_labs["HADM_ID"]==row["HADM_ID"]]
            med2_bool = self.check_medk(t_med2, row)
            lab_vals = lab_vals[lab_vals["LabTimeFromAdmit"].dt.total_seconds()>row["MedTimeFromAdmit"].total_seconds()]
            a_window_start = row["MedTimeFromAdmit"].total_seconds() + (a_w[0]*3600)
            a_window_end = row["MedTimeFromAdmit"].total_seconds() + (a_w[1])*3600
            lab_vals = lab_vals[lab_vals["LabTimeFromAdmit"].dt.total_seconds()>a_window_start]
            lab_vals = lab_vals[lab_vals["LabTimeFromAdmit"].dt.total_seconds()<a_window_end]
            lab_vals["hours_from_med"] = (lab_vals["CHARTTIME"]-row["ENDTIME"]).dt.total_seconds()/3600
            lab_vals = lab_vals.sort_values(["ITEMID", "hours_from_med"])

            if med2_bool:
                med2_val = self.get_medk(t_med2, row)
                lab_vals = lab_vals[lab_vals["LabTimeFromAdmit"].dt.total_seconds()<med2_val["MedTimeFromAdmit"].total_seconds()]
            
            t = lab_vals.groupby(["ITEMID"]).count()[["HADM_ID"]]

            val_counts_m = t[t["HADM_ID"]>=1]
            if val_counts_m.shape[0]==0:
                row[f"after_abs_{a_w}"] = {}
                row[f"after_mean_{a_w}"] = {}
                row[f"after_std_{a_w}"] = {}
                row[f"after_trends_{a_w}"] = {}
                row[f"after_time_{a_w}"] = {}
            else:
                l_m = lab_vals[lab_vals.ITEMID.isin(val_counts_m.index)]
                row[f"after_abs_{a_w}"] = l_m.groupby(["ITEMID"])[["VALUENUM", "hours_from_med"]].first()["VALUENUM"].dropna().to_dict()
                row[f"after_mean_{a_w}"] = l_m.groupby(["ITEMID"])[["VALUENUM"]].mean()["VALUENUM"].dropna().to_dict()
                row[f"after_std_{a_w}"] = l_m.groupby(["ITEMID"])[["VALUENUM"]].std()["VALUENUM"].dropna().to_dict()
                row[f"after_trends_{a_w}"] = l_m[["VALUENUM", "hours_from_med", "ITEMID"]].dropna().groupby(["ITEMID"])[["VALUENUM", "hours_from_med"]].apply(lambda r : get_normalized_trend(r)).dropna().to_dict()
                row[f"after_time_{a_w}"] = l_m.groupby(["ITEMID"])[["VALUENUM", "hours_from_med"]].first()["hours_from_med"].dropna().to_dict()
                
        return row
    
    def generate_med_lab_data(self, before_windows, after_windows, lab_parts=(0,50)):
        """Generate lab test values in before and after windows of medication

        Args:
            before_windows (list of tuples): before windows (in hours) Ex: [(1,2), (2,3)]
            after_windows (list of tuples): after windows (in hours) Ex: [(1,2), (2,3)]
            use_id (bool, optional): 

        Returns:
            list(pd.DataFrame): Med lab pair values are present in this dataframe. Each row contains the medication value and the before/after lab test value.
            list(pd.DataFrame): Contains columns with dictionaries of lab values. Columns are named based on before and after window
        """
        self.final_pairs_data = []
        self.interim_pairs_data = []
        
        t_labs = self.t_labs
        for i in tqdm(range(len(self.meds))):
            
            if i==0:
                t_med0, t_med1, t_med2 = None, self.meds[i], self.meds[i+1]
            elif len(self.meds)!=2 and i==len(self.meds)-1:
                t_med0, t_med1, t_med2 = self.meds[i-1], self.meds[i], None
            else:
                t_med0, t_med1, t_med2 = self.meds[i-1], self.meds[i], self.meds[i+1]
            
            all_types = set(["abs", "mean", "std", "trends", "time"])
            cols_b = [f"before_{t}_{b_w}" for b_w in before_windows for t in all_types]
            cols_a = [f"after_{t}_{a_w}" for a_w in after_windows for t in all_types]
            cols = cols_b.copy()
            cols.extend(cols_a)
            temp = t_med1.copy()

            self.interim_pairs_data.append(temp.progress_apply(lambda r : self.get_vals(r, t_labs, t_med0, t_med2, before_windows, after_windows), axis=1))
            self.interim_pairs_data[-1].to_csv(os.path.join(self.res, f"before_after_windows_main_med_lab_first_val_{self.stratify_prefix}_doc_eval_new_win_lab{lab_parts}_med({i}, {i+1}).csv"))
            temp = self.interim_pairs_data[-1]
            
            col_vals = []
            for col in cols:
                col_vals.append(
                    temp.assign(dict=temp[col].dropna().map(lambda d: json.loads(d.replace("\'", "\"")).items())).explode("dict", ignore_index=True).assign(
                        LAB_ITEMID=lambda df: df.dict.str.get(0),
                        temp=lambda df: df.dict.str.get(1)
                    ).drop(columns=["dict"]+cols).astype({'temp':'float64'}).rename(columns={"temp":f"{col}_sp"}).dropna(subset=["LAB_ITEMID"])
                )
            for i in range(1, len(col_vals)):
                col_vals[i] = pd.merge(col_vals[i-1], col_vals[i], how="outer", on=list(t_med1.columns)+["LAB_ITEMID"])
            
            final = col_vals[-1][list(t_med1.columns)+["LAB_ITEMID"]+[f"{col}_sp" for col in cols]]
            final["LAB_NAME"] = final["LAB_ITEMID"]
            final = final.rename(columns={"ITEMID":"MED_NAME"})
            self.final_pairs_data.append(final)
            
            final.to_csv(os.path.join(self.res, f"before_after_windows_main_med_lab_trends_first_val_{self.stratify_prefix}_doc_eval_win_lab{lab_parts}_med({i}, {i+1}).csv"))

        return self.final_pairs_data, self.interim_pairs_data
    
    def query(self, med, lab, before_windows, after_windows, use_id=False):
        """Query lab test value for a given medication

        Args:
            med (string): Medication name
            lab (string): Lab test name
            before_windows (list of tuples): before windows (in hours) Ex: [(1,2), (2,3)]
            after_windows (list of tuples): after windows (in hours) Ex: [(1,2), (2,3)]
            use_id (bool, optional): Set to True to use Original labels given in MIMIC names. Defaults to False, ie, to use MIMIC Extract labels and HIRID labels.

        Returns:
            list(pd.DataFrame): Med lab pair values are present in this dataframe. Each row contains the medication value and the before/after lab test value.
            list(pd.DataFrame): Contains columns with dictionaries of lab values. Columns are named based on before and after window
        """
        filter_col = constants.ID_COL if use_id else constants.NAME_ID_COL
                
        t_labs =  self.t_labs[self.t_labs[filter_col]==lab]
        n_meds = [med1[med1[filter_col]==med] for med1 in self.meds if med1[med1[filter_col]==med].shape[0]>0]

        t_labs.to_csv(os.path.join(self.res, f't_labs_{lab}.csv'))
        n_meds[0].to_csv(os.path.join(self.res, f'n_meds0_{med}.csv'))

        if n_meds[0].shape[0]==0:
            print(f"No data found for the given medication {med}")
            return
        if t_labs.shape[0]==0:
            print(f"No data found for the given lab test {lab}")
            return
        
        final_pairs_data = []
        interim_pairs_data = []
        
        for i in tqdm(range(len(n_meds))):
            
            if i==0 :
                t_med0, t_med1, t_med2 = None, self.meds[i], self.meds[i+1]
            elif len(self.meds)!=2 and i==len(self.meds)-1:
                t_med0, t_med1, t_med2 = self.meds[i-1], self.meds[i], None
            else:
                t_med0, t_med1, t_med2 = self.meds[i-1], self.meds[i], self.meds[i+1]
            
            all_types = set(["abs", "mean", "std", "trends", "time"])
            cols_b = [f"before_{t}_{b_w}" for b_w in before_windows for t in all_types]
            cols_a = [f"after_{t}_{a_w}" for a_w in after_windows for t in all_types]
            cols = cols_b.copy()
            cols.extend(cols_a)
            temp = t_med1.copy()

            interim_pairs_data.append(temp.progress_apply(lambda r : self.get_vals(r, t_labs, t_med0, t_med2, before_windows, after_windows), axis=1))
            interim_pairs_data[-1].to_csv(os.path.join(self.res, f"before_after_windows_main_med_lab_first_val_{self.stratify_prefix}_doc_eval_new_win_lab-{lab}_med-{med}({i}, {i+1}).csv"))
            temp = interim_pairs_data[-1]
            
            col_vals = []
            for col in cols:
                col_vals.append(
                    temp.assign(dict=temp[col].dropna().map(lambda d: d.items())).explode("dict", ignore_index=True).assign(
                        LAB_ITEMID=lambda df: df.dict.str.get(0),
                        temp=lambda df: df.dict.str.get(1)
                    ).drop(columns=["dict"]+cols).astype({'temp':'float64'}).rename(columns={"temp":f"{col}_sp"}).dropna(subset=["LAB_ITEMID"])
                )
            for i in range(1, len(col_vals)):
                col_vals[i] = pd.merge(col_vals[i-1], col_vals[i], how="outer", on=list(t_med1.columns)+["LAB_ITEMID"])
            
            final = col_vals[-1][list(t_med1.columns)+["LAB_ITEMID"]+[f"{col}_sp" for col in cols]]
            final["LAB_NAME"] = final["LAB_ITEMID"]
            final = final.rename(columns={"ITEMID":"MED_NAME"})
            final_pairs_data.append(final)
            
            final.to_csv(os.path.join(self.res, f"before_after_windows_main_med_lab_trends_first_val_{self.stratify_prefix}_doc_eval_win_lab-{lab}_med-{med}({i}, {i+1}).csv"))

        return final_pairs_data, interim_pairs_data
