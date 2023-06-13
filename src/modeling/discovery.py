import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np


class ClinicalDiscoveryAnalysis:
    def __init__(self, med_lab_pair_data):
        self.med_lab_pair_data = med_lab_pair_data
    
    def statistical_tests(self, med_name, lab_name, before_windows, after_windows, min_patients=100, types_l=["abs"]):
        """Perform statistical tests on the before and after lab test values of given medication and lab test pairs. Comparision done between given before and after windows

        Args:
            med_name (_type_): _description_
            lab_name (_type_): _description_
            before_windows (_type_): _description_
            after_windows (_type_): _description_

        Returns:
            _type_: _description_
        """
        med_lab_data_orig = self.med_lab_pair_data.copy()
        med_lab_data_orig = med_lab_data_orig[med_lab_data_orig["LAB_NAME"]==lab_name]
        med_lab_data_orig = med_lab_data_orig[med_lab_data_orig["MED_NAME"]==med_name]
        
        discovery_res = []
        template = {
            "Lab Name" : lab_name,
            "Med Name": med_name
        }
                
        for t in types_l:
            for aw in after_windows:
                for bw in before_windows:
                    med_lab_data = med_lab_data_orig
                    # Initializing variable for a before and after window
                    row = template.copy()
                    t = "abs"
                    a, b = f"after_{t}_{aw}_sp", f"before_{t}_{bw}_sp"
                    row["Before Window (in Hours)"] = bw
                    row["After Window (in Hours)"] = aw
                    pvals, ttest = [], []
                    med_lab_data = med_lab_data.dropna(subset=[a,b])
                    
                    if med_lab_data.shape[0]>=min_patients:
                        
                        # Performing tests
                        c_m, pval_m = stats.mannwhitneyu(med_lab_data[b], med_lab_data[a])
                        c_t, pval_t = stats.ttest_ind(med_lab_data[b], med_lab_data[a])
                        c_t_p, pval_t_p = stats.ttest_rel(med_lab_data[b], med_lab_data[a])
                        
                        # Adding data to dataframe
                        row["Mannwhitneyu Test"] = pval_m
                        row["TTest Independent"] = pval_t
                        row["TTest Paired"] = pval_t_p
                        row["No of Patients"] = med_lab_data.shape[0]
                        row["Type"] = t
                        discovery_res.append(row)
    
        return discovery_res
    
    def analyze(self, before_windows, after_windows, min_patients=100, types_l=["abs"]):
        """Perform statistical tests to generate p values for all medication<>lab test pairs in the given data

        Args:
            before_windows (_type_): _description_
            after_windows (_type_): _description_

        Returns:
            _type_: _description_
        """
        pairs = self.med_lab_pair_data.groupby(["MED_NAME", "LAB_NAME"]).count().index        
        discovery_res = []
        for med_name, lab_name in pairs:
            res = self.statistical_tests(med_name=med_name, lab_name=lab_name, before_windows=before_windows, after_windows=after_windows, min_patients=min_patients, types_l=types_l)
            if len(res)>0:
                discovery_res.extend(res)
        res_df = pd.DataFrame(discovery_res)
        if "TTest Paired" not in res_df.columns:
            data = res_df.to_dict()
            data_final = [v for k, v in data[0].items() if 'nan' not in v]
            res_df = pd.DataFrame(data_final)
        return res_df
    
    def analyze_ratio(self, before_windows, after_windows, min_patients=10):
        med_lab_pairs = self.med_lab_pair_data
        
        for b_w in before_windows:
            for a_w in after_windows:
                med_lab_pairs[f"ratio_{b_w}_{a_w}"] = med_lab_pairs[f"after_abs_{a_w}_sp"] / med_lab_pairs[f"before_abs_{b_w}_sp"]
        
        if sorted(after_windows)[0][0]==0 and sorted(before_windows)[0][0]==0:
            a_w = sorted(after_windows)[0]
            b_w = sorted(before_windows)[0]
            t = med_lab_pairs.dropna(subset=[f"ratio_{b_w}_{a_w}"])
            if t[t[f"ratio_{b_w}_{a_w}"]==1].shape[0]>100:
                med_lab_pairs = pd.concat([ med_lab_pairs[(med_lab_pairs[f"after_time_{a_w}_sp"]<1) & (med_lab_pairs[f"ratio_{b_w}_{a_w}"]!=1)], med_lab_pairs[(med_lab_pairs[f"after_time_{a_w}_sp"]>=1) | (med_lab_pairs[f"after_time_{a_w}_sp"].isna())] ])
        
        med_lab_pairs.groupby(["MED_NAME", "LAB_NAME"]).count()[["HADM_ID"]]
        pairs_df = med_lab_pairs.groupby(["MED_NAME", "LAB_NAME"]).count()[["HADM_ID"]]
        pairs = pairs_df[pairs_df["HADM_ID"]>100].index
        
        discovery_res1 = []
        for med_name, lab_name in pairs:
            stat_test_df = []
            for a_w in after_windows:
                for b_w in before_windows:
                    vals = med_lab_pairs[med_lab_pairs["LAB_NAME"]==lab_name]
                    vals = vals[vals["MED_NAME"]==med_name]
                    vals = vals[f"ratio_{b_w}_{a_w}"].replace([np.inf, -np.inf], np.nan).dropna()
                    if vals.shape[0]>min_patients:
                        res = stats.ttest_1samp(vals.to_numpy(), popmean=1)
                        row = {
                            "Lab Name": lab_name,
                            "Med Name": med_name,
                            "Before Window (in Hours)": b_w,
                            "After Window (in Hours)": a_w,
                            "No. of Patients": vals.shape[0],
                            "1-Sampled Ttest" : res.pvalue
                        }
                        stat_test_df.append(row)
            if len(stat_test_df)>0:
                discovery_res1.extend(stat_test_df)
        
        res_df = pd.DataFrame(discovery_res1)
        # res_df[res_df["No. of Patients"]>min_patients]
        return res_df
    
    def generate_significant(self, pvals_med_lab, alpha=0.01, statistical_test="TTest Paired"):
        """Choose significant medication<>lab test pairs using Bonferroni and FDR analysis with pvals from the given statistical test

        Args:
            pvals_med_lab (_type_): _description_
            statistical_test (str, optional): _description_. Defaults to "Ttest Paired".

        Returns:
            _type_: _description_
        """
        
        test_pval_data = pvals_med_lab.copy().reset_index().drop(columns=["index"])
        
        # pvals - Bonferrroni Analysis
        bonferroni_analysis = multipletests(test_pval_data[statistical_test], alpha=alpha, method='bonferroni')
        reject_bonferroni, pvals_corrected, _, alphacBonf = bonferroni_analysis
        test_pval_data["BonferroniPvals"] = pd.Series(pvals_corrected)

        ### pvals - FDR Analysis
        fdr1_analysis = multipletests(test_pval_data[statistical_test], alpha=alpha, method='fdr_bh')
        reject_fdr, pvals_corrected1, _, alphacBonf = fdr1_analysis
        test_pval_data['FDR Benjamini Corrected'] = pd.Series(pvals_corrected1)

        # choose significant
        significant_hard_thres = test_pval_data[test_pval_data[statistical_test]<alpha]
        significant_bonferroni = test_pval_data[reject_bonferroni]
        significant_fdr = test_pval_data[reject_fdr]
        
        return test_pval_data, significant_hard_thres, significant_bonferroni, significant_fdr