import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import pearsonr, spearmanr

from src.utils.utils import AnalysisUtils, remove_outlier, plot_func, plot_corrs


class ClinicalPlotAnalysis(AnalysisUtils):

    def __init__(self, data, res, gender="MF", age_b=0, age_a=100, ethnicity="WHITE", lab_mapping=None):
       super().__init__(data, res, gender=gender, age_b=age_b, age_a=age_a, ethnicity=ethnicity, lab_mapping=lab_mapping)
    
    def make_plot_dirs(self):
        """
        Create folders to store output plots
        """
        plot_dir = os.path.join(self.res, f"plots_{self.stratify_prefix}_doc_eval")
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)
        plot_dir1 = os.path.join(self.res, f"plots_{self.stratify_prefix}_doc_eval_all_window")
        if not os.path.isdir(plot_dir1):
            os.mkdir(plot_dir1)
        return plot_dir, plot_dir1
    
    def generate_plot_data(self, final, before_windows, after_windows):
        """
        Generate plot data for each before and after window
        """
        t_final = final.copy()
        plot_data = {}

        # generate column names
        all_types = set(["abs", "time"])
        cols_b = [f"before_{t}_{b_w}" for b_w in before_windows for t in all_types]
        cols_a = [f"after_{t}_{a_w}" for a_w in after_windows for t in all_types]
        
        # get data for each before and after window
        for b in [f"{c}_sp" for c in cols_b]:
            if b in t_final.columns:
                plot_data[b] = []
                for a in [f"{c}_sp" for c in cols_a]:
                    if a in t_final.columns:
                        plot_data[b].append(t_final.dropna(subset=[a,b]))
        plot_data_concat = {}
        for i in plot_data:
            plot_data_concat[i] = pd.concat(plot_data[i])

        # Generate columns names
        a_t = ["abs", "time"]
        cols_b_sp = [(f"before_{a_t[0]}_{b_w}_sp", f"before_{a_t[1]}_{b_w}_sp") for b_w in before_windows]
        cols_a_sp = [(f"after_{a_t[0]}_{a_w}_sp", f"after_{a_t[1]}_{a_w}_sp") for a_w in after_windows]
        cols_sp = cols_b_sp.copy()
        cols_sp.extend(cols_a_sp)
        
        # get data for each before and after window
        t_final = final.copy()
        plot_data = {}
        for b in cols_b_sp:
            if b[0] in t_final.columns:
                plot_data[b[0]] = {}
                for a in cols_a_sp: 
                    if a[0] in t_final.columns:
                        plot_data[b[0]][a[0]] = t_final.dropna(subset=[a[0], a[1], b[0], b[1]])
        pickle.dump(plot_data, open(f"plot_bw_aw_med_lab_data_{self.stratify_prefix}_doc_eval_win.pkl", "wb"))
        
        # get data for each medication<>labtest pair
        cols_d = dict(cols_sp)
        p_data = {}
        for k in plot_data:
            for i, (k_a, data) in enumerate(plot_data[k].items()):
                
                t_data = data.set_index([data["MED_NAME"], data["LAB_NAME"]])        
                med_lab_pairs = t_data.index
                
                for med_lab_pair in med_lab_pairs.unique():
                    
                    if med_lab_pair not in p_data.keys():
                        p_data[med_lab_pair] = {}
                    if k not in p_data[med_lab_pair].keys():
                        p_data[med_lab_pair][k] = []
                    
                    t_d = t_data.loc[med_lab_pair][['SUBJECT_ID','HADM_ID', k, k_a, cols_d[k_a]]]
                    t_d['abs'] = t_d[k_a]-t_d[k]
                    t_d['percent'] = (t_d['abs']/t_d[k])*100
                    t_d['ratio'] = t_d[k_a]/t_d[k]
                    t_d.replace([np.inf, -np.inf], np.nan, inplace=True)
                    t_d = t_d.dropna()
                    p_data[med_lab_pair][k].append(t_d)

        pickle.dump(p_data, open(f"plot_med_lab_bw_aw_data_{self.stratify_prefix}_doc_eval_win.pkl", "wb"))
        return p_data
        

    def plot(self, final, t_labs, before_windows, after_windows):
        """
        Plots the correlation and trend plots for before and after windows. Returns a data frame with medication labtest pairs and change over time.
        Three different type of plots:
        1. Correlation plot (across all after windows for a before window) - plots_corr
        2. Large trend/data plot (across all after windows for a before window) - plot_func (type 1)
        3. Smaller trend/data plots (for each after window and before window) - plot_func (type 2)
        """

        plot_dir, plot_dir1 = self.make_plot_dirs()
        type_map = {
            'abs': "Absolute",
            'percent': "Percentage",
            'ratio': "Ratio"
        }

        med_vals_new, labtest_vals_new = self.generate_med_lab_pairs()

        lab_units_mapping = t_labs.groupby(["ITEMID", "VALUEUOM"]).count()["SUBJECT_ID"].reset_index().groupby("ITEMID").nth(0)[["VALUEUOM"]]
        lab_units_mapping_dict = lab_units_mapping.to_dict()['VALUEUOM']

        p_data = self.generate_plot_data(final, before_windows, after_windows)
        
        n_p_data = {}
        if len([i for i in zip(med_vals_new, labtest_vals_new)]) < len(p_data):
            for k in [i for i in zip(med_vals_new, labtest_vals_new)]:
                if k in p_data:
                    n_p_data[k] = p_data[k]
        old_p_data = p_data.copy()
        p_data = n_p_data

        before_windows_map = {f"({str(b_w)[1:-1]})":b_w for b_w in before_windows}
        after_windows_map = {f"({str(a_w)[1:-1]})":a_w for a_w in after_windows}
        
        # For each medication and lab test pair plot before and after window correlations and trends
        types = ['abs', 'percent', 'ratio']
        type2 = ""
        stratify_prefix = self.stratify_prefix
        corrs_data_dict = []
        for k, v in p_data.items():
            for key in v:
                if "/" in k[0]:
                    presc = k[0].split("/")[0]
                else:
                    presc = k[0]
                lab = k[1]
                before_window = before_windows_map[key.split("_")[-2]]

                fig_all, ax_all = plt.subplots(3, figsize=(20, 20))
                fig_all.suptitle(f'{lab}<>{presc} (before window = {str(before_window)})')

                if not os.path.isdir(os.path.join(plot_dir, f"{lab}<>{presc}")):
                    os.mkdir(os.path.join(plot_dir, f"{lab}<>{presc}"))
                
                dirname=f"bw_{before_window}"
                if not os.path.isdir(os.path.join(plot_dir, f"{lab}<>{presc}", dirname)):
                    os.mkdir(os.path.join(plot_dir, f"{lab}<>{presc}", dirname))
                
                #  Iterating over the typw of analysis : ["Absolute", "Percentage", "Ratio"]
                for i, type1 in enumerate(types):
                    
                    # Get data type2, remove outliers and calculate correlation
                    plot_name = f"{lab}<>{presc}_{key}_{type1}"
                    data_vals = [d[[list(d.columns)[-4], type1]].rename(columns={list(d.columns)[-4] : "time"}) for d in v[key] if type(d) != pd.Series]
                    after_names = [list(d.columns)[3] for d in v[key] if type(d) != pd.Series]
                    type2 = type1
                    if len(data_vals)!=len(after_names):
                        print(data_vals)
                        print(after_names)
                        print()
                        continue
                    if len(data_vals)==0:
                        continue
                    d = pd.concat(data_vals)
                    if d.shape[0]<2:
                        continue
                    if d.shape[0]>1:
                        d1, d2 = remove_outlier(d[type2], d["time"])
                    else:
                        d1, d2 = d[[type2]], d[["time"]]
                    d = pd.concat([d1, d2], axis=1)

                    #  Calculate correlation ovver all after windows
                    p_corr = pearsonr(d1[type2], d2["time"])
                    s_corr = spearmanr(d1[type2], d2["time"])

                    # Get units for the plot and plot overall data plot
                    unit = lab_units_mapping_dict[lab] if lab in lab_units_mapping_dict else ""
                    plot_func(lab, presc, d[[type2, "time"]].rename(columns={type2:"data"}), dirname="", labels=(p_corr, s_corr), plot_dir=plot_dir, plot_dir1=plot_dir1, unit=unit, title=f"bw{before_window} {type_map[type2]}", plot_name=f"{plot_name}", ax=ax_all[i])

                    # Correlation plots 
                    fig_corrs, ax_corrs = plt.subplots(2, figsize=(20, 20))
                    fig_corrs.suptitle(f'{lab}<>{presc} {type2} corrs') 
                    corrs = []
                    data_t = []
                    temp_after_names = after_names.copy()
                    for i, d in enumerate(data_vals):
                        if d.shape[0]<2:
                            temp_after_names.remove(after_names[i])
                            continue
                        if d.shape[0]>1:
                            d1, d2 = remove_outlier(d[type2], d["time"])
                        else:
                            d1, d2 = d[[type2]], d[["time"]]
                        p_corr = pearsonr(d1[type2], d2["time"])
                        s_corr = spearmanr(d1[type2], d2["time"])
                        corrs.append((p_corr, s_corr))
                        data_t.append([d1, d2])
                    after_names = temp_after_names
                    plot_corrs(corrs, after_names, after_windows_map, ax_corrs, title=type2, plot_name=plot_name)
                    
                    # Make correlation plot with all windows
                    fig_corrs.savefig(os.path.join(plot_dir, f"{lab}<>{presc}", dirname, f"{plot_name}_{type2}_{stratify_prefix}_corrs.png"))
                    fig_corrs.clf()

                    for d, a, c, t in zip(data_vals, after_names, corrs, data_t):
                        d = pd.concat(t, axis=1)
                        p_corr = c[0]
                        s_corr = c[1]
                        after_window = after_windows_map[a.split("_")[-2]]

                        # Make plot with data points in a single window
                        plot_func(lab, presc, d[[type2, "time"]].rename(columns={type2:"data"}), dirname=dirname, plot_dir=plot_dir, plot_dir1=plot_dir1, labels=c, unit=unit, title=f"bw{before_window} aw{after_window} {type_map[type2]}", plot_name=f"{plot_name} bw{before_window} aw{after_window}")
                        corrs_data_dict.append({
                            "lab" : lab,
                            "med": presc,
                            "bw": before_window,
                            "aw": after_window,
                            "Type": type_map[type2],
                            "Pearson Correlation": p_corr[0],
                            "Pearson Correlation (p-value)": p_corr[1],
                            "Spearmans Correlation ": s_corr[0],
                            "Spearmans Correlation (p-value)": s_corr[1],
                            "Num of Data Points (n)": d.shape[0]
                        })

                fig_all.savefig(os.path.join(plot_dir, f"{lab}<>{presc}", dirname, f"{plot_name}_{stratify_prefix}.png"))
                fig_all.clf()
        
        # Save correlation values for each type, after and before window
        corrs_data_df = pd.DataFrame(corrs_data_dict)
        corrs_data_df.to_csv(os.path.join(plot_dir, f"corrs_data_{stratify_prefix}.csv"))
        return corrs_data_df
