import pandas as pd

from src.utils.utils import AnalysisUtils


class DatasetQuerier(AnalysisUtils):

    def __init__(self, data, res, gender="MF", age_b=0, age_a=100, ethnicity="WHITE", lab_mapping=None):
        self.final = None
        self.temp = None
        super().__init__(data, res, gender=gender, age_b=age_b, age_a=age_a, ethnicity=ethnicity, lab_mapping=lab_mapping)
    
    def check_med2(self, t_med2, row):
        """
        Check if a 2nd medication was administered to patients
        """
        if row["HADM_ID"] in t_med2["HADM_ID"].to_list():
            if row["ITEMID"] in t_med2[t_med2["HADM_ID"]==row["HADM_ID"]]["ITEMID"].to_list():
                return True
        return False

    def get_med2(self, t_med2, row):
        '''
        Return 2nd medication data
        '''
        temp = t_med2[t_med2["HADM_ID"]==row["HADM_ID"]] 
        return temp[temp["ITEMID"]==row["ITEMID"]].iloc[0]
    
    def get_vals(self, r, t_labs, t_med1, t_med2, before_windows, after_windows):
        """
        Calculate the lab test values in time windows before and after medication administration. Return a dataframe with labtest values of before and after windows as a dict
        Params: 
        - before_windows: list of tuples (each tuple is a window)
        - after_windows: list of tuples (each tuple is a window)
        """

        row = r.copy()
        for b_w in before_windows:
            lab_vals = t_labs[t_labs["HADM_ID"]==row["HADM_ID"]]
            lab_vals = lab_vals[lab_vals["LabTimeFromAdmit"].dt.total_seconds()<row["MedTimeFromAdmit"].total_seconds()]

            b_window_start = row["MedTimeFromAdmit"].total_seconds() - (b_w[0]*3600)
            b_window_end = row["MedTimeFromAdmit"].total_seconds() - (b_w[1])*3600
            lab_vals = lab_vals[lab_vals["LabTimeFromAdmit"].dt.total_seconds()<b_window_start]
            lab_vals = lab_vals[lab_vals["LabTimeFromAdmit"].dt.total_seconds()>b_window_end]
            lab_vals["hours_from_med"] = (row["STARTTIME"]-lab_vals["CHARTTIME"]).dt.total_seconds()/3600
            lab_vals = lab_vals.sort_values(["ITEMID", "hours_from_med"])

            t = lab_vals.groupby(["ITEMID"]).count()[["HADM_ID"]]
            val_counts_m = t[t["HADM_ID"]>=1]
            if val_counts_m.shape[0]==0:
                row[f"before_abs_{b_w}"] = {}
                row[f"before_mean_{b_w}"] = {}
                row[f"before_trends_{b_w}"] = {}
                row[f"before_time_{b_w}"] = {}
            else:
                l_m = lab_vals[lab_vals.ITEMID.isin(val_counts_m.index)]
                row[f"before_abs_{b_w}"] = l_m.groupby(["ITEMID"])[["VALUENUM", "hours_from_med"]].first()["VALUENUM"].dropna().to_dict()
                row[f"before_mean_{b_w}"] = l_m.groupby(["ITEMID"])[["VALUENUM"]].mean()["VALUENUM"].dropna().to_dict()
                row[f"before_trends_{b_w}"] = l_m[["VALUENUM", "hours_from_med", "ITEMID"]].dropna().groupby(["ITEMID"])[["VALUENUM", "hours_from_med"]].apply(lambda r : get_normalized_trend(r)).dropna().to_dict()
                row[f"before_time_{b_w}"] = l_m.groupby(["ITEMID"])[["VALUENUM", "hours_from_med"]].first()["hours_from_med"].dropna().to_dict()

        for a_w in after_windows:

            lab_vals = t_labs[t_labs["HADM_ID"]==row["HADM_ID"]]
            med2_bool = self.check_med2(t_med2, row)
            lab_vals = lab_vals[lab_vals["LabTimeFromAdmit"].dt.total_seconds()>row["MedTimeFromAdmit"].total_seconds()]
            a_window_start = row["MedTimeFromAdmit"].total_seconds() + (a_w[0]*3600)
            a_window_end = row["MedTimeFromAdmit"].total_seconds() + (a_w[1])*3600
            lab_vals = lab_vals[lab_vals["LabTimeFromAdmit"].dt.total_seconds()>a_window_start]
            lab_vals = lab_vals[lab_vals["LabTimeFromAdmit"].dt.total_seconds()<a_window_end]
            lab_vals["hours_from_med"] = (lab_vals["CHARTTIME"]-row["ENDTIME"]).dt.total_seconds()/3600
            lab_vals = lab_vals.sort_values(["ITEMID", "hours_from_med"])

            if med2_bool:
                med2_val = self.get_med2(t_med2, row)
                lab_vals = lab_vals[lab_vals["LabTimeFromAdmit"].dt.total_seconds()<med2_val["MedTimeFromAdmit"].total_seconds()]
            
            t = lab_vals.groupby(["ITEMID"]).count()[["HADM_ID"]]
            
            val_counts_m = t[t["HADM_ID"]>=1]
            if val_counts_m.shape[0]==0:
                row[f"after_abs_{a_w}"] = {}
                row[f"after_mean_{a_w}"] = {}
                row[f"after_trends_{a_w}"] = {}
                row[f"after_time_{a_w}"] = {}
            else:
                l_m = lab_vals[lab_vals.ITEMID.isin(val_counts_m.index)]
                row[f"after_abs_{a_w}"] = l_m.groupby(["ITEMID"])[["VALUENUM", "hours_from_med"]].first()["VALUENUM"].dropna().to_dict()
                row[f"after_mean_{a_w}"] = l_m.groupby(["ITEMID"])[["VALUENUM"]].mean()["VALUENUM"].dropna().to_dict()
                row[f"after_trends_{b_w}"] = l_m[["VALUENUM", "hours_from_med", "ITEMID"]].dropna().groupby(["ITEMID"])[["VALUENUM", "hours_from_med"]].apply(lambda r : get_normalized_trend(r)).dropna().to_dict()
                row[f"after_time_{a_w}"] = l_m.groupby(["ITEMID"])[["VALUENUM", "hours_from_med"]].first()["hours_from_med"].dropna().to_dict()
                
        return row
    
    def generate_med_lab_data(self, t_labs, t_med1, t_med2, before_windows, after_windows):
        """
        Generate lab test values in before and after windows of medication
        """
        
        all_types = set(["abs", "time"])
        cols_b = [f"before_{t}_{b_w}" for b_w in before_windows for t in all_types]
        cols_a = [f"after_{t}_{a_w}" for a_w in after_windows for t in all_types]
        cols = cols_b.copy()
        cols.extend(cols_a)
        temp = t_med1.copy()

        temp = temp.apply(lambda r : self.get_vals(r, t_labs, t_med1, t_med2, before_windows, after_windows), axis=1)
        self.temp = temp
        temp.to_csv(self.res, f"before_after_windows_main_med_lab_first_val_{self.stratify_prefix}_doc_eval_new_win.csv")
        
        col_vals = []
        for col in cols:
            col_vals.append(
                temp.assign(dict=temp[col].map(lambda d: d.items())).explode("dict", ignore_index=True).assign(
                    LAB_ITEMID=lambda df: df.dict.str.get(0),
                    temp=lambda df: df.dict.str.get(1)
                ).drop(columns=["dict"]+cols).astype({'temp':'float64'}).rename(columns={"temp":f"{col}_sp"}).dropna(subset=["LAB_ITEMID"])
            )
        for i in range(1, len(col_vals)):
            col_vals[i] = pd.merge(col_vals[i-1], col_vals[i], how="outer", on=list(t_med1.columns)+["LAB_ITEMID"])
        
        final = col_vals[-1][list(t_med1.columns)+["LAB_ITEMID"]+[f"{col}_sp" for col in cols]]
        final["LAB_NAME"] = final["LAB_ITEMID"]
        final = final.rename(columns={"ITEMID":"MED_NAME"})
        self.final = final
        
        final.to_csv(self.res, f"before_after_windows_main_med_lab_trends_first_val_{self.stratify_prefix}_doc_eval_win.csv")

        return final, temp
    
    def query(self):
        """
        Query lab test value for a given medication
        """
        pass
