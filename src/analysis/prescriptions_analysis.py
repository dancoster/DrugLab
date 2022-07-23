import pandas as pd
import datetime
import random
import numpy as np
from scipy.stats import mannwhitneyu
from scipy import stats
from tqdm import tqdm
import os
import gzip
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics
import logging
from analysis.analysis import Analysis

logging.basicConfig(level=logging.INFO, format=f'%(module)s/%(filename)s [Class: %(name)s Func: %(funcName)s] %(levelname)s : %(message)s')

class PRDataAnalysis(Analysis):

    def __init__(self, path, dataset):
        self.logger = logging.getLogger(self.__class__.__name__)
        Analysis.__init__(self, path, dataset, 'prescriptions')