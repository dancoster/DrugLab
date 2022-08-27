import pandas as pd
import datetime
import numpy as np

from scipy.stats import mannwhitneyu
from scipy import stats
from statsmodels.stats.multitest import multipletests

from tqdm import tqdm
import os


class SignificantPairs:

    def __init__(self, stats_test, suffix=''):
        self.stats_test = stats_test
        self.suffix=f'_{suffix}'
        self.enum = {
            'absolute' : ['Absolute-', self.absolute_significant],
            'interpolated' : ['', self.interpolated_significant],
            'trend' : ['Coef-', self.trends_significant],
        }
        self.enum_stats = {
            'mannwhitney':'Mannwhitney-pvalue',
            'ttest':'Ttest-pvalue'
        }
        self.enum_stats_opp = {
            'mannwhitney':'Ttest-pvalue',
            'ttest':'Mannwhitney-pvalue'
        }
    
    def get_name(self, test_type):
        return f'{self.enum[test_type][0]}{self.enum_stats[self.stats_test]}'
    
    def get_name_opp(self, test_type):
        return f'{self.enum[test_type][0]}{self.enum_stats_opp[self.stats_test]}'
        
    def bonferroni(self, pvals, res_analysis):
        bonferroni_analysis = multipletests(pvals, alpha=0.05, method='bonferroni')
        reject, pvals_corrected, _, alphacBonf = bonferroni_analysis
        res_analysis['Bonferroni Corrected'] = pvals_corrected
        significant = res_analysis[reject]
        return significant

    def fdr1_benjamini(self, pvals, res_analysis):
        fdr1_analysis = multipletests(pvals, alpha=0.05, method='fdr_bh')
        reject1, pvals_corrected1, _, alphacBonf = fdr1_analysis
        res_analysis['FDR Benjamini Corrected'] = pvals_corrected1
        significant_fdr = res_analysis[reject1]
        return significant_fdr
    
    def absolute_significant(self, res_analysis, test_type, res_path):
        pvals = res_analysis[self.get_name(test_type)]
        merged = pd.merge(self.bonferroni(pvals, res_analysis), self.fdr1_benjamini(pvals, res_analysis), how='inner').drop(
                columns=['Lab Test Before(std)', 'Time Before(std)', 'Lab Test After(std)', 'Time After(std)', 'Ttest-pvalue', 'Estimated (std)','Estimated (mean)', 'Mannwhitney-pvalue',	'Before',	'After',	'Coef-Ttest-pvalue'	,'Coef-Mannwhitney-pvalue']
            ).sort_values(
                ['Number of patients'], ascending=False
        )
        merged = merged.sort_values(['Number of patients'], ascending=False).sort_values(['FDR Benjamini Corrected', 'Bonferroni Corrected'])
        merged.to_csv(f'{res_path[:-4]}{self.suffix}_significant_{self.stats_test}_{test_type}.csv')
        return merged
    
    def interpolated_significant(self, res_analysis, test_type, res_path):
        pvals = res_analysis[self.get_name(test_type)]
        merged = pd.merge(self.bonferroni(pvals, res_analysis), self.fdr1_benjamini(pvals, res_analysis), how='inner').drop(
                columns=['Lab Test After(std)', 'Time After(std)', 'Absolute-Ttest-pvalue',	'Absolute-Mannwhitney-pvalue',	'Before',	'After',	'Coef-Ttest-pvalue'	,'Coef-Mannwhitney-pvalue']
        )
        merged = merged.sort_values(['Number of patients'], ascending=False).sort_values(['FDR Benjamini Corrected', 'Bonferroni Corrected'])
        merged.to_csv(f'{res_path[:-4]}{self.suffix}_significant_{self.stats_test}_{test_type}.csv')
        l = pd.merge(self.bonferroni(pvals, res_analysis), self.fdr1_benjamini(pvals, res_analysis), how='inner').drop(
                columns=['Lab Test After(std)', 'Time After(std)', 'Absolute-Ttest-pvalue',	'Absolute-Mannwhitney-pvalue',	'Before',	'After',	'Coef-Ttest-pvalue'	,'Coef-Mannwhitney-pvalue']
        ).sort_values(['Number of patients'], ascending=False).sort_values(['FDR Benjamini Corrected', 'Bonferroni Corrected'])
        l.to_csv(f'{res_path[:-4]}{self.suffix}_significant_{self.stats_test}_{test_type}.csv')
        return l
    
    def trends_significant(self, res_analysis, test_type, res_path):
        res_analysis = res_analysis[['Medication', 'Lab Test', 'Coef-Mannwhitney-pvalue', 'Coef-Ttest-pvalue', 'Before', 'After',
       'Number of patients']]
        pvals = res_analysis[self.get_name(test_type)]
        merged = pd.merge(self.bonferroni(pvals, res_analysis), self.fdr1_benjamini(pvals, res_analysis), how='inner').sort_values(
            ['Number of patients'], ascending=False
        )
        merged = merged.sort_values(['Number of patients'], ascending=False).sort_values(['FDR Benjamini Corrected', 'Bonferroni Corrected'])
        merged.to_csv(f'{res_path[:-4]}{self.suffix}_significant_{self.stats_test}_{test_type}.csv')
        return merged

    def get_significant_pairs(self, res_analysis, test_type, res_path):
        merged = self.enum[test_type][1](res_analysis, test_type, res_path)
        return merged