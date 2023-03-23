import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import pandas as pd


class ClinicalDiscoveryAnalysis:
    def __init__(self, med_lab_pair_data):
        self.med_lab_pair_data = med_lab_pair_data
    
    def statistical_tests(self, med_name, lab_name, before_windows, after_windows):
        """Perform statistical tests on the before and after lab test values of given medication and lab test pairs. Comparision done between given before and after windows

        Args:
            med_name (_type_): _description_
            lab_name (_type_): _description_
            before_windows (_type_): _description_
            after_windows (_type_): _description_

        Returns:
            _type_: _description_
        """
        med_lab_data = self.med_lab_pair_data.copy()
        med_lab_data = med_lab_data[med_lab_data["LAB_NAME"]==lab_name]
        med_lab_data = med_lab_data[med_lab_data["MED_NAME"]==med_name]
        
        discovery_res = []
        template = {
            "Lab Name" : lab_name,
            "Med Name": med_name
        }

        for aw in after_windows:
            for bw in before_windows:
                # Initializing variable for a before and after window
                row = template.copy()
                t = "abs"
                a, b = f"after_{t}_{aw}_sp", f"before_{t}_{bw}_sp"
                row["Before Window (in Hours)"] = bw
                row["After Window (in Hours)"] = aw
                pvals, ttest = [], []
                med_lab_data = med_lab_data.dropna(subset=[a,b])
                
                if med_lab_data.shape[0]==0:
                    row["Mannwhitneyu Test"] = 1
                    row["TTest Independent"] = 1
                    row["TTest Paired"] = 1
                    row["No of Patients"] = 0
                    discovery_res.append(row)
                    continue
                
                # Performing tests
                c_m, pval_m = stats.mannwhitneyu(med_lab_data[b], med_lab_data[a])
                c_t, pval_t = stats.ttest_ind(med_lab_data[b], med_lab_data[a])
                c_t_p, pval_t_p = stats.ttest_rel(med_lab_data[b], med_lab_data[a])
                
                # Adding data to dataframe
                row["Mannwhitneyu Test"] = pval_m
                row["TTest Independent"] = pval_t
                row["TTest Paired"] = pval_t_p
                row["No of Patients"] = med_lab_data.shape[0]
                discovery_res.append(row)
    
        return discovery_res
    
    def analyze(self, before_windows, after_windows):
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
            res = self.statistical_tests(med_name=med_name, lab_name=lab_name, before_windows=before_windows, after_windows=after_windows)
            if len(res)>0:
                discovery_res.append(res)
        res_df = pd.DataFrame(discovery_res)
        return res_df
    
    def generate_significant(self, pvals_med_lab, alpha=0.01, statistical_test="TTest Paired"):
        """Choose significant medication<>lab test pairs using Bonferroni and FDR analysis with pvals from the given statistical test

        Args:
            pvals_med_lab (_type_): _description_
            statistical_test (str, optional): _description_. Defaults to "Ttest Paired".

        Returns:
            _type_: _description_
        """
        
        test_pval_data = pvals_med_lab.copy()
        
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