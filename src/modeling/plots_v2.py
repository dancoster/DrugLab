import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from src.utils.utils import AnalysisUtils, get_normalized_trend
from src.utils import constants


class ClinicalPlotsAnalysis(AnalysisUtils):

    def __init__(self, data, res, med_lab_pairs, plot_dir="plots", gender="MF", age_b=0, age_a=100, ethnicity="WHITE", lab_mapping=None):
        """Class used to make plots

        Args:
            data (_type_): _description_
            res (_type_): _description_
            med_lab_pairs (_type_): _description_
            gender (str, optional): _description_. Defaults to "MF".
            age_b (int, optional): _description_. Defaults to 0.
            age_a (int, optional): _description_. Defaults to 100.
            ethnicity (str, optional): _description_. Defaults to "WHITE".
            lab_mapping (_type_, optional): _description_. Defaults to None.
        """
        self.med_lab_pairs = med_lab_pairs
        self.plot_dir = plot_dir
        super().__init__(data, res, gender=gender, age_b=age_b, age_a=age_a, ethnicity=ethnicity, lab_mapping=lab_mapping)
    
    def generate_data(self, med_name, lab_name, start_time_before=4, start_time_after=8, min_interval_between_tests=1, min_human=20, max_human=1000):
        """Generate data to plot

        Args:
            med_name (_type_): _description_
            lab_name (_type_): _description_
            start_time_before (int, optional): _description_. Defaults to 4.
            start_time_after (int, optional): _description_. Defaults to 8.
            min_interval_between_tests (int, optional): _description_. Defaults to 1.
            min_human (int, optional): _description_. Defaults to 20.
            max_human (int, optional): _description_. Defaults to 1000.

        Returns:
            pd.DataFrame: _description_
        """
        drug_name = med_name
        xx = self.med_lab_pairs
        
        # take only patients with insulin and glucose
        yy = xx[(xx.MED_NAME == drug_name) & (xx.LAB_NAME == lab_name)]

        # take only subjects with measurements from the 3 hours before drug administartion TODO - bug
        yy = yy[(yy['before_time_(0, 12)_sp'] >= 0) & (yy['before_time_(0, 12)_sp'] <= start_time_after)]

        # exclude out patients with no before or after TODO - bug
        yy = yy[(~yy['after_abs_(0, 12)_sp'].isna()) & (~yy['before_abs_(0, 12)_sp'].isna())]

        #remove inhuman values from before & after values
        yy = yy[(yy['after_abs_(0, 12)_sp'] > min_human) & (yy['after_abs_(0, 12)_sp'] < max_human)]
        yy = yy[(yy['before_abs_(0, 12)_sp'] > min_human) & (yy['before_abs_(0, 12)_sp'] < max_human)]

        #exclude testes with interval less than one hour
        yy = yy[(yy['before_time_(0, 12)_sp'] + yy['after_time_(0, 12)_sp']) > min_interval_between_tests]

        # Exclude measurements up to 20 minutes after the test
        #yy = yy[yy['after_time_(0, 12)_sp'] >0.26]

        # Take only up to 4 hours
        yy = yy[yy['after_time_(0, 12)_sp'] <= start_time_after]

        # Sample n=30 individuals per 0.5h after_time interval
        #yy['after_time_(0, 12)_sp'] = round_off_rating(yy['after_time_(0, 12)_sp'])
        #yy['after_time_(0, 12)_sp'] = round(yy['after_time_(0, 12)_sp'],0)

        #import random
        #temp_seed = random.randint(1,1000)
        #n_samples = min(yy['after_time_(0, 12)_sp'].value_counts())
        #print('n samples' + str(n_samples))
        #yy = yy.groupby("after_time_(0, 12)_sp").sample(n=n_samples, random_state=temp_seed)

        #yy[['SUBJECT_ID','before_abs_(0, 12)_sp','before_time_(0, 12)_sp', 'after_abs_(0, 12)_sp','after_time_(0, 12)_sp', 'LAB_NAME']]
        #calculate the ratio
        yy['ratio'] = yy['after_abs_(0, 12)_sp']/yy['before_abs_(0, 12)_sp']
        yy = yy.sort_values(by=['before_abs_(0, 12)_sp'])
        
        return yy
    
    def plot_reg(self, med_name, lab_name, start_time_before=4, start_time_after=8, min_interval_between_tests=1, min_human=20, max_human=1000):
        """Plot regression plots with distribution

        Args:
            preprocessed_data (_type_): _description_
        """
        yy = self.generate_data(med_name, lab_name, start_time_before=start_time_before, start_time_after=start_time_after, min_interval_between_tests=min_interval_between_tests, min_human=min_human, max_human=max_human)
        if yy.shape[0]<2:
            print(f"No Data for {med_name}<>{lab_name} pair")
            return
        
        graph = sns.jointplot(data=yy,x='after_time_(0, 12)_sp', y='ratio', kind='reg',scatter_kws={"color": "black"}, line_kws={"color": "red"})
        r, p = stats.pearsonr(yy['after_time_(0, 12)_sp'],yy['ratio'])
        # if you choose to write your own legend, then you should adjust the properties then
        phantom, = graph.ax_joint.plot([], [], linestyle="", alpha=0)
        # here graph is not a ax but a joint grid, so we access the axis through ax_joint method
        graph.ax_joint.legend([phantom],['r={:f}, p={:f}'.format(r,p)])
        sns.regplot(x=yy['after_time_(0, 12)_sp'], y=yy['ratio'], scatter_kws={"color": "black"}, line_kws={"color": "red"})
        
        # save and clean plot
        if "/" in med_name:
            med_name = med_name[:med_name.index("/")]
        if "/" in lab_name:
            lab_name = lab_name[:lab_name.index("/")]
        plt.savefig(os.path.join(self.plot_dir, f"{lab_name}<>{med_name}_{start_time_after}_{start_time_before}_{min_interval_between_tests}_{min_human}_{max_human}_reg_plot.png"))
        plt.clf()
        
    def plot_med_lab(self, med_name, lab_name, start_time_before=4, start_time_after=8, min_interval_between_tests=1, min_human=20, max_human=1000):
        """Plot med<>lab plots

        Args:
            med_name (_type_): _description_
            lab_name (_type_): _description_
            start_time_before (int, optional): _description_. Defaults to 4.
            start_time_after (int, optional): _description_. Defaults to 8.
            min_interval_between_tests (int, optional): _description_. Defaults to 1.
            min_human (int, optional): _description_. Defaults to 20.
            max_human (int, optional): _description_. Defaults to 1000.
        """
        yy = self.generate_data(med_name, lab_name, start_time_before=start_time_before, start_time_after=start_time_after, min_interval_between_tests=min_interval_between_tests, min_human=min_human, max_human=max_human)
        if yy.shape[0]<2:
            print(f"No Data for {med_name}<>{lab_name} pair")
            return
        
        #calc eps
        min_range = min(pd.concat([yy['before_abs_(0, 12)_sp'],yy['after_abs_(0, 12)_sp']]))
        max_range= max(pd.concat([yy['before_abs_(0, 12)_sp'],yy['after_abs_(0, 12)_sp']]))
        eps = 0.05*(max_range-min_range)

        #calc line size
        len_before = (start_time_before/(start_time_after + start_time_before))
        len_after = (start_time_after/(start_time_after + start_time_before))

        plt.figure(figsize=(6, 3), dpi=300)
        plt.xlim([(-1)*start_time_before, start_time_after])
        plt.plot(yy['before_time_(0, 12)_sp']*-1,yy['before_abs_(0, 12)_sp'], 'o')
        plt.axhline(y=yy['before_abs_(0, 12)_sp'].median(),xmin = 0.03, xmax = len_before - 0.03,color = 'blue')
        plt.text((start_time_before/2)*(-1), yy['before_abs_(0, 12)_sp'].median()+ eps, str(round(yy['before_abs_(0, 12)_sp'].median(),2)), horizontalalignment='center', size='medium', color='black', weight='semibold',bbox=dict(boxstyle="round",facecolor='white', edgecolor='blue'))

        plt.plot(yy['after_time_(0, 12)_sp'],yy['after_abs_(0, 12)_sp'], 'o')
        plt.axhline(y=yy['after_abs_(0, 12)_sp'].median(),xmin = len_before+0.03, xmax = 0.97,color = 'orange') 
        plt.text(start_time_after/2, yy['after_abs_(0, 12)_sp'].median()+ eps, str(round(yy['after_abs_(0, 12)_sp'].median(),2)), horizontalalignment='center', size='medium', color='black', weight='semibold',bbox=dict(boxstyle="round",facecolor='white', edgecolor='orange'))

        plt.axvline(x=0,color='grey')
        
        if "/" in med_name:
            med_name = med_name[:med_name.index("/")]
        if "/" in lab_name:
            lab_name = lab_name[:lab_name.index("/")]
        plt.savefig(os.path.join(self.plot_dir, f"{lab_name}<>{med_name}_{start_time_after}_{start_time_before}_{min_interval_between_tests}_{min_human}_{max_human}_med_lab_plot.png"))
        plt.clf()
        
    def plot_all(self, start_time_before=4, start_time_after=8, min_interval_between_tests=1, min_human=20, max_human=1000):
        """Plot both regression plots and med<>lab plots for all choosen med<>lab pairs

        Args:
            start_time_before (int, optional): _description_. Defaults to 4.
            start_time_after (int, optional): _description_. Defaults to 8.
            min_interval_between_tests (int, optional): _description_. Defaults to 1.
            min_human (int, optional): _description_. Defaults to 20.
            max_human (int, optional): _description_. Defaults to 1000.
        """
        pairs = self.med_lab_pairs.groupby(["MED_NAME", "LAB_NAME"]).count().index
        for med_name, lab_name in pairs:
            self.plot_reg(med_name, lab_name, start_time_before=start_time_before, start_time_after=start_time_after, min_interval_between_tests=min_interval_between_tests, min_human=min_human, max_human=max_human)
            self.plot_med_lab(med_name, lab_name, start_time_before=start_time_before, start_time_after=start_time_after, min_interval_between_tests=min_interval_between_tests, min_human=min_human, max_human=max_human)
