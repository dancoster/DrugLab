# K-fold module: Train ML models and evaluate performance using K-fold validation
import pandas as pd
import pickle
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm
from src.imputation import ml_models
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pygam import ExpectileGAM
import scipy
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve, accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

def Kfold_splits(df, Kfold):
    """  Get split indices """
    ids_list = df['subject_id'].unique()
    splits_zipped = []
    kf = KFold(n_splits=Kfold)

    for train_index, test_index in kf.split(ids_list):
        test_ids = ids_list[test_index].tolist()
        train_ids = ids_list[train_index].tolist()

        test_indexes = df.index[df['subject_id'].isin(test_ids)].tolist()
        train_indexes = df.index[df['subject_id'].isin(train_ids)].tolist()
        splits_zipped.append([test_indexes, train_indexes])

    return splits_zipped

def is_data_standardized(df, features):
    """  Check if the data is already standardized """
    epsilon = 1e-10
    for feature in features:
        mean = df[feature].mean()
        std = df[feature].std()
        # check if mean == 0 and std == 1 (up to epsilon)
        if not ((0 - epsilon <= mean) and (mean <= 0 + epsilon) and
                (1 - epsilon <= std) and (std <= 1 + epsilon)):
            print("The values of %s are not standardized: mean=%.3f, std=%.3f, epsilon=%f" % (
                feature, mean, std, epsilon))
            return False
    return True

def z_score_norm_seen(df, features):
    """  Standardize values of seen_data (TRAINING set). """
    features_params = {}
    for feature in features:
        std = df[feature].std()
        mean = df[feature].mean()
        if std == 0:  # if the std is 0, change nothing
            features_params[feature] = {'mean': 0,
                                        'std': 1}
        else:
            df[feature] = (df[feature] - mean) / std
            features_params[feature] = {'mean': mean,
                                        'std': std}
    return df, features_params


def z_score_norm_unseen(df, features, train_params):
    """ Standardize values of unseen_data (TEST set).
        It uses the mean and std of the TRAINING explanatory variables.
        In this way, we can test whether our model can generalize well to new, unseen data points.
    """
    for feature in features:
        std = train_params[feature]['std']
        mean = train_params[feature]['mean']
        df[feature] = (df[feature] - mean) / std
    return df

def time_series_imputation(df, features, method='fill_forward'):
    """ Perform imputation for time series missing values, by the given method """
    if method == 'fill_forward':
        df[features] = df.groupby('subject_id')[features].transform(lambda x: x.fillna(method='ffill'))
        
    elif method == 'linear_interpolation':
        interpolation_func = lambda x: x.interpolate(method='linear', limit_direction='forward', axis=0)
        df[features] = df.groupby('subject_id')[features].transform(interpolation_func)

    # else - doesn't perform imputation
    return df

def benchmark_imputation(X_train, y_train, X_test, imputation_method, features,df_drug_forward_imputed, numerical_cols,
                         categorical_cols, models_vector, save_coef, logfile):
    """ Perform imputation and run different ML model on the imputed dataset.
        Returns DF containing models performance results """ 
    write_log(logfile, f"\n** Perform {imputation_method} imputation")
    
    X_train_before_imputation = X_train.copy()

    X_train_before_imputation.to_csv('X_train_before_imputation'+imputation_method+'.csv')

    if imputation_method == 'drug_forward':
        X_train = df_drug_forward_imputed.loc[X_train.index]
        X_test = df_drug_forward_imputed.loc[X_test.index]

    else:
        X_train = time_series_imputation(X_train, numerical_cols, imputation_method)
        X_test = time_series_imputation(X_test, numerical_cols, imputation_method)

    X_train.to_csv('X_Train_'+imputation_method+'.csv')

    # impute first values per feature with mean
    X_train[numerical_cols] = X_train[numerical_cols].fillna(X_train_before_imputation[numerical_cols].mean())
    X_test[numerical_cols] = X_test[numerical_cols].fillna(X_train_before_imputation[numerical_cols].mean())

    X_train.to_csv('X_Train_include_first_values'+imputation_method+'.csv')

    # Remove index cols
    X_train = X_train[features]
    X_test = X_test[features]
    print("Final features: ", list(X_train.columns))

    lists_of_features = {"features": features,
                         "categorical": categorical_cols}

    # run models as specified in the models vector
    model_results = ml_models.run_models(X_train, y_train, X_test, models_vector, save_coef, lists_of_features)
    model_results["imputation_method"] = imputation_method
    
    return model_results

def impute_drug_forward(df, drug_forward_params,logfile, features):
#def impute_drug_forward(df, temp_df_meds,df_inhuman,delta_time_before,delta_time_after,mimic_data_querier,inputevents_mv,df_d_items):

    #extract paramerters from dictionary
    delta_time_after = drug_forward_params['delta_time_after']
    delta_time_before = drug_forward_params['delta_time_before']
    mimic_data_querier = drug_forward_params['mimic_data_querier']
    inputevents_mv = drug_forward_params['inputevents_mv']
    df_d_items = drug_forward_params['df_d_items']
    df_inhuman = drug_forward_params['df_inhuman']
    temp_df_meds = drug_forward_params['temp_df_meds']
    drug_forward_type = drug_forward_params['drug_forward_type']

    df_drugs_counts = pd.DataFrame(columns=['drug_name','lab_name','pat_counter','adms_counter','imputed_value','total_null_obs'])

    for lab_name in temp_df_meds['Lab Name'].unique():
        # Extract relevant lab_name
        med_t_name = temp_df_meds[temp_df_meds['Lab Name'] == lab_name]['Med Name'].to_list()[0]


        # Extract only inputevents that related to the drugs associated with lab_name, df_med includes them
        df_inputevents,df_med = extract_med_per_lab(lab_name,temp_df_meds,df_d_items,inputevents_mv)

        if (med_t_name == lab_name):
            new_med_t_name = med_t_name+'temp_drug'
            df_med.loc[df_med['med_label'] == med_t_name,df_med.columns == 'med_label'] = new_med_t_name
            med_t_name = new_med_t_name

        # add cols of adminstration of each of this drugs to df
        df_drugs = add_med_adminstrations_cols(df_inputevents,df,df_med)

        # save file of subjects who got this drug
        df_drugs.to_csv(med_t_name+'.csv')

        if (drug_forward_type == 'single_drug'):
            print('Create function for '+lab_name+ '<>'+med_t_name)

            # Calculate gam function
            # gam_func_temp =  create_gum_func(df_inhuman,lab_name,med_t_name,delta_time_before,delta_time_after,mimic_data_querier)
            gam_func_temp = create_gum_func_df(df_inhuman,lab_name,med_t_name,delta_time_before,delta_time_after,mimic_data_querier)

            #impute missing values using gam functions
            df_temp_masked_imputed = drug_forward_imputation(df_drugs.copy(),lab_name,med_t_name,delta_time_before,gam_func_temp,delta_time_after,drug_forward_type)

        if (drug_forward_type == 'multi_drugs'):
            #calculate estimated value per medication
            for med_t_name in df_med.med_label:
                print('Create function for '+lab_name+ '<>'+med_t_name)
                #gam_func_temp = create_gum_func(df_inhuman,lab_name,med_t_name,delta_time_before,delta_time_after,mimic_data_querier)
                gam_func_temp = create_gum_func_df(df_inhuman,lab_name,med_t_name,delta_time_before,delta_time_after,mimic_data_querier)
                df_drugs = drug_forward_imputation(df_drugs,lab_name,med_t_name,delta_time_before,gam_func_temp,delta_time_after,drug_forward_type)
                df_drugs.to_csv('df_drugs_temp'+med_t_name+'.csv')

            df_drugs.to_csv('df_drugs_temp.csv')

            df_temp_masked_imputed = df_drugs.copy()
            df_temp_masked_imputed.loc[:,df_temp_masked_imputed.columns==lab_name] = df_temp_masked_imputed[[lab_name+'_'+med_t_name for med_t_name in df_med.med_label.unique()]].mean(axis=1)

        for med_name in df_med.med_label.unique():
            adms_counter = (len(df_drugs[~df_drugs[med_name].isna()]))
            pat_counter = (len(df_drugs[~df_drugs[med_name].isna()]['subject_id'].unique()))
            imputed_values = (df_temp_masked_imputed[lab_name].isna().value_counts()[1] - df[lab_name].isna().value_counts()[1])
            lab_col_len = (df[lab_name].isna().value_counts()[0])
            print('For '+lab_name+ ' using the med '+med_t_name+': '+str(imputed_values)+' NaN values were imputed using drug_forward')

            tempRow = [med_name,lab_name,pat_counter,adms_counter,imputed_values, lab_col_len]

            write_log(logfile, f"\n! ! ! med={med_name}, lab={lab_name}, asdms={adms_counter}, patients={pat_counter}")

            df_drugs_counts = df_drugs_counts.append(pd.Series(tempRow, index=df_drugs_counts.columns), ignore_index=True)

        df_temp_masked_imputed.to_csv('df_temp_masked_imputed_'+lab_name+'.csv')

        # assign imputed values
        df_temp_masked_imputed.sort_index(inplace=True)
        df = df.sort_index().reset_index(drop = True)

        df_temp_masked_imputed.to_csv('df_temp_masked_imputed_after_'+lab_name+'.csv')
        df.to_csv('df_after_'+lab_name+'.csv')

        df[lab_name] = df_temp_masked_imputed[lab_name]

    df_drugs_counts.to_csv('df_drugs_counts.csv')

    #ffill the rest of the features
    df[features] = df.groupby('subject_id')[features].transform(lambda x: x.fillna(method='ffill'))

    return(df)

def compare_models_Kfold_validation(df, models_vector, imputation_methods,
                                    features,drug_forward_params, n_folds=10,
                                    run_standardization=False,
                                    save_coef=False, reports_path="", logfile=""):
    """ Train and compare ML models using K-fold CV.
        Returns performance results """

    splits_zipped = Kfold_splits(df, n_folds)
    risk_scores_df = pd.DataFrame()
    
    X = df.drop(columns=['target'])
    y = df['target']
    X.to_pickle(reports_path + f"X.pkl")
    y.to_pickle(reports_path + f"y.pkl")

    index_cols = ['target', 'subject_id', 'charttime']
    categorical_cols = list(X.columns[X.dtypes == 'bool'])
    numerical_cols = [col for col in X.columns if col not in categorical_cols and \
                      not col=='subject_id' and \
                      not col=='charttime']
    print("Categorical variables: ", categorical_cols)
    print("Numerical variables: ", numerical_cols)
    assert sorted(numerical_cols + categorical_cols + index_cols) == sorted(df.columns), "Error! Additional column should be dropped."
    assert sorted(numerical_cols + categorical_cols) == sorted(features), "Error! A feature is potentially missing."

    #create imputed drug_forward dataset
    if (('drug_forward') in (imputation_methods)):
        df_drug_forward_imputed = impute_drug_forward(X, drug_forward_params,logfile,features)
        #convert to preic
        df_drug_forward_imputed.index = X.index
        df_drug_forward_imputed.to_csv('df_drug_forward_imputed.csv')

    split_dict = {}
    split = 1

    for test_index, train_index in tqdm(splits_zipped):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

        split_dict[split] = {'train' : train_index, 
                             'test' : test_index}
        write_log(logfile, f"\n! ! ! fold={split}: Train={X_train.shape}, Test={X_test.shape}")

        # Standartization
        if run_standardization: # required for part of the imputation methods
            write_log(logfile, "\n* Perform standartization\n")
            X_train, train_params = standardize_data(X_train, numerical_cols)
            X_test = standardize_unseen_data(X_test, numerical_cols, train_params)

        # Imputation with most frequent for categorial values, and mean/median/knn/mice for numerical
        # A. Imputate Categorical
        if len(categorical_cols) > 0:
            X_train[categorical_cols] = X_train[categorical_cols].fillna(X_train[categorical_cols].mode().iloc[0])
            X_test[categorical_cols] = X_test[categorical_cols].fillna(X_train[categorical_cols].mode().iloc[0])

        # B. Numerical imputation benchmark
        for imputation_method in imputation_methods:
            model_results = benchmark_imputation(X_train.copy(), y_train.copy(), X_test.copy(),
                                               imputation_method,
                                               features, df_drug_forward_imputed, numerical_cols, categorical_cols,
                                               models_vector,
                                               save_coef=save_coef, logfile=logfile)
            model_results["Split"] = split
            model_results["Target"] = y_test
            x = pd.DataFrame.from_dict(model_results)
            risk_scores_df = risk_scores_df.append(x, sort=False)
        split += 1
    
    risk_scores_df.to_pickle(reports_path + 'risk_scores.pkl')
    with open(reports_path + 'split_index.pkl', 'wb') as f:
        pickle.dump(split_dict, f)
    
    return risk_scores_df


def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b

# https://swharden.com/blog/2020-09-24-python-exponential-fit/
def calc_exponent_func(x,y,lab_name,med_t_name):
    xs = np.array(x)
    ys = np.array(y)

    plt.plot(xs, ys, '.')
    plt.title("Original Data")

    # perform the fit
    p0 = (.1, .1, 10) # start with values near those we expect
    params, cv = scipy.optimize.curve_fit(monoExp, xs, ys, p0)
    m, t, b = params
    sampleRate = 20_000 # Hz
    tauSec = (1 / t) / sampleRate

    # determine quality of the fit
    squaredDiffs = np.square(ys - monoExp(xs, m, t, b))
    squaredDiffsFromMean = np.square(ys - np.mean(ys))
    rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
    print(f"R² = {rSquared}")

    # plot the results
    plt.plot(xs, ys, '.', label="data")
    x_axis_values = np.linspace(min(xs), max(xs), num=500)
    plt.plot(x_axis_values, monoExp(x_axis_values, m, t, b), '--', label="exponent")
    plt.title("Fitted Exponential Curve "+f"\n R² = {round(rSquared,3)}" + (f" | Y = {round(m,3)} * e^(-{round(t,3)} * x) + {round(b,3)}"))
    plt.legend()
    plt.axhline(y = 1, color = 'r', linestyle = '-')
    plt.xlabel("Hours since "+med_t_name+ " adminstration")
    plt.ylabel('Ratio of ' + lab_name)
    plt.savefig('exponential_'+lab_name+'_'+med_t_name+'.jpeg')
    plt.clf()
    # inspect the parameters
    print(f"Y = {m} * e^(-{t} * x) + {b}")
    print(f"Tau = {tauSec * 1e6} µs")

    return(params)

def extract_ratio_and_after_time(df_drug_lab, min_val, max_val,delta_time_before,delta_time_after):
    # remove inhuman values
    df_drug_lab = df_drug_lab[df_drug_lab['after_abs_(0, 12)_sp'] > min_val]
    df_drug_lab = df_drug_lab[df_drug_lab['before_abs_(0, 12)_sp'] > min_val]
    df_drug_lab = df_drug_lab[df_drug_lab['after_abs_(0, 12)_sp'] <max_val]
    df_drug_lab = df_drug_lab[df_drug_lab['before_abs_(0, 12)_sp'] <max_val]

    # take data only according to time windows
    df_drug_lab = df_drug_lab[df_drug_lab['before_time_(0, 12)_sp'] <= int(delta_time_before[0])]
    df_drug_lab = df_drug_lab[df_drug_lab['after_time_(0, 12)_sp'] <= int(delta_time_after[0])]

    #calculate ratio
    df_drug_lab['ratio'] = df_drug_lab['after_abs_(0, 12)_sp']/df_drug_lab['before_abs_(0, 12)_sp']
    print(df_drug_lab.shape)

    #remove ratio == 1 because of lab_events and chartevents bug
    df_drug_lab = df_drug_lab[~((df_drug_lab['ratio'] == 1) & (df_drug_lab['after_time_(0, 12)_sp']<=1))]

    #remove rows where one of the values is mising
    df_drug_lab = df_drug_lab[~df_drug_lab['ratio'].isna()]
    df_drug_lab = df_drug_lab[~df_drug_lab['ratio'].isna()]
    print(df_drug_lab.shape)

    #define vectors for GAM calculation
    x = df_drug_lab['after_time_(0, 12)_sp']
    y = df_drug_lab['ratio']

    return(x,y)

# https://pygam.readthedocs.io/en/latest/notebooks/tour_of_pygam.html
def calc_gam_func(x,y,lab_name,med_t_name):
    X = np.array([[i] for i in list(x)])
    y =  pd.DataFrame(y)

    # lets fit the mean model first by CV
    gam50 = ExpectileGAM(expectile=0.5).gridsearch(X, y)

    XX = gam50.generate_X_grid(term=0, n=500)
    plt.title('GAM')
    plt.scatter(X, y, c='k', alpha=0.2)
    plt.axhline(y = 1, color = 'r', linestyle = '-')
    plt.xlabel("Hours since "+med_t_name+ " adminstration")
    plt.ylabel('Ratio of ' + lab_name)
    plt.plot(XX, gam50.predict(XX), label='GAM')
    plt.legend()
    plt.savefig('gam_'+lab_name+'_'+med_t_name+'.jpeg')
    plt.clf()
    print('GAM was created')
    return(gam50)

#drug forward
def drug_forward_imputation(df,lab_name,med_t_name,delta_time_before,gam_func,delta_time_after,drug_forward_type):
    #create empty col for drugs
    #df[lab_name+'_'+med_t_name] = np.nan

    for pat_id in tqdm(df.subject_id.unique()):
        # calculate imputed value per subject
        df_pat = preprocessed_data_per_id(df, pat_id,lab_name,med_t_name,delta_time_after,int(delta_time_before[0]))
        imputed_vec = fill_drug_forward_values(df_pat, med_t_name,gam_func.predict,lab_name,delta_time_after)

        # assign imputed_vec
        if (drug_forward_type == 'single_drug'):
            df.loc[df.subject_id == pat_id, lab_name] = imputed_vec.to_list()
        # assign imputed_vec
        if (drug_forward_type == 'multi_drugs'):
            df.loc[df.subject_id == pat_id, lab_name+'_'+med_t_name] = imputed_vec.to_list()

    return(df)



def extract_ratio_and_after_time_df(df_drug_lab, min_val, max_val,delta_time_before,delta_time_after):
    # remove inhuman values
    df_drug_lab = df_drug_lab[df_drug_lab['after_abs_(0, 4)_sp'] > min_val]
    df_drug_lab = df_drug_lab[df_drug_lab['before_abs_(0, 6)_sp'] > min_val]
    df_drug_lab = df_drug_lab[df_drug_lab['after_abs_(0, 4)_sp'] <max_val]
    df_drug_lab = df_drug_lab[df_drug_lab['before_abs_(0, 6)_sp'] <max_val]

    # take data only according to time windows
    df_drug_lab = df_drug_lab[df_drug_lab['before_time_(0, 6)_sp'] <= int(delta_time_before[0])]
    df_drug_lab = df_drug_lab[df_drug_lab['after_time_(0, 4)_sp'] <= int(delta_time_after[0])]

    #calculate ratio
    df_drug_lab['ratio'] = df_drug_lab['after_abs_(0, 4)_sp']/df_drug_lab['before_abs_(0, 6)_sp']
    print(df_drug_lab.shape)

    #remove ratio == 1 because of lab_events and chartevents bug
    df_drug_lab = df_drug_lab[~((df_drug_lab['ratio'] == 1) & (df_drug_lab['after_time_(0, 4)_sp']<=1))]

    #remove rows where one of the values is mising
    df_drug_lab = df_drug_lab[~df_drug_lab['ratio'].isna()]
    df_drug_lab = df_drug_lab[~df_drug_lab['ratio'].isna()]
    print(df_drug_lab.shape)

    #define vectors for GAM calculation
    x = df_drug_lab['after_time_(0, 4)_sp']
    y = df_drug_lab['ratio']

    return(x,y)
def create_gum_func_df(df_inhuman,lab_name,med_t_name,delta_time_before,delta_time_after,final):
    max_val = float(df_inhuman[df_inhuman.full_name == lab_name]['max_inhuman'].tolist()[0])
    min_val = float(df_inhuman[df_inhuman.full_name == lab_name]['min_inhuman'].tolist()[0])
    print(max_val)
    print(min_val)

    b_w = [(0,6)]
    a_w = [(0,4)]

    #load_data_
    med_lab_pair = final[(final.LAB_NAME == lab_name) & (final['MED_NAME'] == med_t_name)]
    df_drug_lab = med_lab_pair
    print(df_drug_lab.shape)

    x,y = extract_ratio_and_after_time_df(df_drug_lab, min_val, max_val,delta_time_before,delta_time_after)
    exp_params, gam_func = exponent_gam_plot(x,y,lab_name,med_t_name)
    #exp_params = calc_exponent_func
    #gam_func = calc_gam_func(x,y,lab_name,med_t_name)

    return(gam_func)

def create_gum_func(df_inhuman,lab_name,med_t_name,delta_time_before,delta_time_after,mimic_data_querier):
    max_val = float(df_inhuman[df_inhuman.full_name == lab_name]['max_inhuman'].tolist()[0])
    min_val = float(df_inhuman[df_inhuman.full_name == lab_name]['min_inhuman'].tolist()[0])

    b_w = [(0,12)]
    a_w = [(0,12)]

    #load_data_
    med_lab_pair = mimic_data_querier.query(med_t_name, lab_name, b_w, a_w)
    df_drug_lab = med_lab_pair[0]

    x,y = extract_ratio_and_after_time(df_drug_lab, min_val, max_val,delta_time_before,delta_time_after)
    exp_params, gam_func = exponent_gam_plot(x,y,lab_name,med_t_name)
    #exp_params = calc_exponent_func
    #gam_func = calc_gam_func(x,y,lab_name,med_t_name)

    return(gam_func)


def extract_med_per_lab(lab_name,df_meds,df_d_items,inputevents_mv):
    #Create lab specific list of medications
    df_med = df_meds[(df_meds['Lab Name'] == lab_name)][['Lab Name','Med Name']].rename(columns={'Med Name':'med_label'})

    #pull drug itemid from d_items
    med_item_id = df_d_items[(df_d_items.LABEL.isin(df_med.med_label)) & (df_d_items.DBSOURCE == 'metavision')][['LABEL','ITEMID']]

    # add item id to dataframe
    df_med = df_med.merge(med_item_id, left_on='med_label', right_on='LABEL').drop(['LABEL'], axis=1)

    #pull relevant medications to lab_name
    df_inputevents = inputevents_mv[inputevents_mv.ITEMID.isin(df_med.ITEMID)]

    # round start time to hour
    df_inputevents['STARTTIME_rounded'] = pd.to_datetime(df_inputevents['STARTTIME'], utc=True).dt.round(freq='H')

    #Take mean amount or drug per hour
    df_inputevents= df_inputevents.groupby(['ITEMID','SUBJECT_ID','STARTTIME_rounded'])['AMOUNT'].mean().reset_index()

    return(df_inputevents,df_med)


def add_med_adminstrations_cols(df_inputevents,df_data,df_med):

    #Add drug cols
    for temp_item_id in df_med.ITEMID:
        temp_df_inputevents = df_inputevents[df_inputevents.ITEMID == temp_item_id].reset_index()
        df_data = pd.merge(df_data, temp_df_inputevents,  how='left', left_on=['subject_id','charttime'], right_on = ['SUBJECT_ID','STARTTIME_rounded'])
        df_data = df_data.drop(['SUBJECT_ID','STARTTIME_rounded','index','ITEMID'], axis=1)
        med_name = df_med[df_med.ITEMID == temp_item_id]['med_label'].iloc[0]
        df_data = df_data.rename(columns={"AMOUNT": med_name})

    # sort charttime in descending order per subject id
    df_data = df_data.sort_values(['subject_id','charttime'])

    return(df_data.copy())

def preprocessed_data_per_id(df_lab_data, pat_id,lab_name,med_t_name,delta_time_after,delta_time_before):
    # filter data per sample
    df_pat = df_lab_data[(df_lab_data.subject_id == pat_id)][['charttime','subject_id',lab_name]+[med_t_name]]

    #convert chartime to index
    df_pat.index = df_pat.charttime

    # count how many dosages were adminstered on the next delta_time_after(=4) hours.
    df_pat[med_t_name+'_counter'] = df_pat[med_t_name].rolling(delta_time_after).count()

    # Create delta_time_after(=4) cols for estimated value
    for i in range(int(delta_time_after[0])):
        df_pat['drug'+'_'+str(i+1)] = np.full(len(df_pat[med_t_name]), np.nan)

    #create new col 'new'_labname with ffil forward lab_name up to delta_time_before(=6) window
    df_pat = fffil_deltatime_feature(df_pat,lab_name,delta_time_before)

    #reset charttime(index)
    df_pat = df_pat.reset_index(drop=True)

    return(df_pat)

def fill_drug_forward_values(df_pat, med_t_name,temp_GAM,lab_name,delta_time_after):
    int_delta_time_after = int(delta_time_after[0])

    #add counter per number of values of lab name
    df_pat['indictor'] = df_pat[lab_name]
    df_pat.loc[~df_pat[lab_name].isna(),'indictor'] = 1
    df_pat['counter'] = df_pat.groupby('indictor').cumcount() + 1
    df_pat['counter'] = df_pat['counter'].ffill()

    if (df_pat[lab_name].count() == 0):
        df_pat['counter'] = 0

    # iterate over all drug cols
    for ind in range(len(df_pat[med_t_name])):

        #check if drug was adminstered
        if (~np.isnan(df_pat[med_t_name][ind])):

            # the counter of lab values in the time of drug adminstration
            lab_counter = df_pat['counter'][ind]
            if (~np.isnan(lab_counter)):
                last_value_current_lab_ind = max(df_pat[df_pat['counter'] == lab_counter].index)
            else:
                last_value_current_lab_ind = np.nan

            #calculate estimate lab result
            temp_vec = ([round(temp_GAM(i)[0]*df_pat['new_'+lab_name][ind],2) for i in range(0,int_delta_time_after)])

            # assign estimated value based on previous values
            assigned = 0
            for i in range(1,int_delta_time_after+1):

                # check one of the drug cols is empty
                if ((np.isnan(df_pat['drug_'+str(i)][ind])) & (assigned == 0)):

                    #claculate length of values to be assigned (created in order to handle values in the last observation of the subject)
                    vec_len = len(df_pat.loc[df_pat.index.isin(list(range(ind,ind+4))).tolist(),'drug_'+str(i)])

                    if ((last_value_current_lab_ind - ind) < 3):
                        vec_len = (last_value_current_lab_ind - ind) + 1

                    #assign values
                    df_pat.loc[df_pat.index.isin(list(range(ind,ind+vec_len))).tolist(),'drug_'+str(i)] = temp_vec[0:vec_len]

                    assigned+=1

    # Claculated imputed values based on drug effected glucose
    df_pat[lab_name+'d_forward'] = round(df_pat[[('drug_'+str(i)) for i in range(1,int_delta_time_after+1)]].mean(axis=1),2)

    #assign imputed values for NaN Glucose values
    df_pat.loc[df_pat[lab_name].isna(),lab_name] = df_pat[df_pat[lab_name].isna()][lab_name+'d_forward']

    return (df_pat[lab_name])

def deltaDates(x, y):
    """ Returns the time difference between two dates, in days """
    return ((x - y).dt.seconds / (60 * 60 * 24) + ((x - y).dt.days))


def add_time_from_last_measurement(df, features):
    """ Given a data-frame and set of features, generate for each feature a new column representing
        the time from the last measurement (in hours) """
    cols_to_drop = []
    new_columns_ls = []
    for feature in features:
        prev_date_col = feature + "_" + "prev_date"
        interval_prev_date_col = 'interval_' + prev_date_col
        cols_to_drop.append(prev_date_col)
        new_columns_ls.append(interval_prev_date_col)
        df[prev_date_col] = df.charttime.where(~np.isnan(df[feature]))
        df[prev_date_col] = df.groupby('subject_id')[prev_date_col].shift()
        df[prev_date_col] = df.groupby('subject_id')[prev_date_col].transform(lambda x: x.fillna(method='ffill'))
        df[interval_prev_date_col] = round(deltaDates(df.charttime, df[prev_date_col]) * 24, 2)
    df = df.drop(columns=cols_to_drop)

    return df, new_columns_ls

def fffil_deltatime_feature(df_data_med,lab_name,delta_time_before):
    temp_new_lab_name = 'new_'+lab_name

    df_data_med['diff_time'] = df_data_med.groupby('subject_id')['charttime'].diff() / np.timedelta64(1, 'h')

    res = add_time_from_last_measurement(df_data_med,[lab_name])
    df = res[0]
    interval_time_lab_name = res[1][0]

    # change to zero if value was measured at that time
    df.loc[list(~df[lab_name].isna()), list((interval_time_lab_name == df.columns))] = 0

    #replicate col
    df[temp_new_lab_name]= df[lab_name]

    # fill forward all values
    df[temp_new_lab_name] = df.groupby('subject_id')[temp_new_lab_name].ffill()

    #convert to NaN Values that were measured after delta_time_before=4h
    df.loc[list((df[interval_time_lab_name]>delta_time_before)), list((temp_new_lab_name == df.columns))] = np.NaN

    return(df.copy())

def write_log(logfile, msg):
    print(msg)
    with open(logfile, 'a') as f:
        f.write(msg)

def calc_risk_scores(risk_scores_df,models_vector,ver):
    df_performance = pd.DataFrame(columns=['Split','imputation_method','model','auc','aupr'])

    for ind_split in risk_scores_df['Split'].unique():
        for imp_method in risk_scores_df['imputation_method'].unique():
            for model in list(models_vector.keys()):
                y_prob = risk_scores_df[(risk_scores_df['Split']==ind_split) & (risk_scores_df['imputation_method']==imp_method)][model]
                y_target = risk_scores_df[(risk_scores_df['Split']==ind_split) & (risk_scores_df['imputation_method']==imp_method)]['Target']

                fpr, tpr, thresholds = roc_curve(y_target, y_prob, drop_intermediate=False)
                aucRes = roc_auc_score(y_target, y_prob)

                precision, recall, thresholds_pr = precision_recall_curve(y_target, y_prob)
                aupr = auc(recall, precision)

                tempRow = [ind_split,imp_method,model,aucRes, aupr]
                df_performance = df_performance.append(pd.Series(tempRow, index=df_performance.columns), ignore_index=True)

    df_performance.to_csv('df_performance_'+ver+'.csv')

    return(df_performance)

def comparison_table(df_performance,metric,ver):
    res = df_performance.groupby(['imputation_method','model'])[metric].median().reset_index().sort_values(by='model').rename(columns={metric: 'median_'+metric})
    res['median_'+metric] = round(res['median_'+metric],4)*100
    mean = round(df_performance.groupby(['imputation_method','model'])[metric].mean().reset_index().sort_values(by='model').rename(columns={metric: 'mean_'+metric})['mean_'+metric],4)*100
    sd = round(df_performance.groupby(['imputation_method','model'])[metric].std().reset_index().sort_values(by='model').rename(columns={metric: 'std'})['std'],4)*100

    res_summary = pd.concat([res,mean,sd], axis=1)

    #res_summary['new'] = str(res_summary['median_auc'])+u"\u00B1"+str(res_summary['std'])
    res_summary['median_std'] = res_summary['median_'+metric].apply("{:.02f}".format).astype(str)+u"\u00B1"+res_summary['std'].apply("{:.02f}".format).astype(str)
    res_summary['mean_std'] = res_summary['mean_'+metric].apply("{:.02f}".format).astype(str)+u"\u00B1"+res_summary['std'].apply("{:.02f}".format).astype(str)

    df_mean_std = res_summary[['imputation_method','model','mean_std']]
    df_mean_std = df_mean_std.pivot(index='imputation_method', columns='model', values='mean_std').reset_index()
    df_mean_std.to_csv('df_mean_std_'+metric+'_'+ver+'.csv', encoding="utf-8-sig")

    df_median_std = res_summary[['imputation_method','model','median_std']]
    df_median_std = df_median_std.pivot(index='imputation_method', columns='model', values='median_std').reset_index()
    df_median_std.to_csv('df_median_std_'+metric+'_'+ver+'.csv', encoding="utf-8-sig")

    print(f"\n comparison table of {metric} was generated")

def generate_comparison_plot(metric,df_performance,ver):
    sns.boxplot(y=metric, x='model',
                     data=df_performance,
                     palette="colorblind",
                     hue='imputation_method')
    plt.legend(bbox_to_anchor=(1.04, 1.08), loc=1, borderaxespad=0.)

    plt.savefig("aupr_"+metric+'_'+ver+'.png')
    plt.show()

def exponent_gam_plot(x,y,lab_name,med_t_name):
    plt.title(lab_name+" & "+med_t_name)

    ## Exponent ##
    xs = np.array(x)
    ys = np.array(y)

    # perform the expnonent fit
    p0 = (.1, .1, 10) # start with values near those we expect
    exp_params, cv = scipy.optimize.curve_fit(monoExp, xs, ys, p0)
    m, t, b = exp_params
    print(f"Y = {m} * e^(-{t} * x) + {b}")

    # determine quality of the fit
    squaredDiffs = np.square(ys - monoExp(xs, m, t, b))
    squaredDiffsFromMean = np.square(ys - np.mean(ys))
    rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
    print(f"R² = {rSquared}")

    ## GAM ##
    X_gam = np.array([[i] for i in list(x)])
    y_gam =  pd.DataFrame(y)

    # lets fit the mean model first by CV
    gam50 = ExpectileGAM(expectile=0.5).gridsearch(X_gam, y_gam)

    # plot the results
    plt.plot(xs, ys, '.')

    #plot exponent
    x_axis_values = np.linspace(min(xs), max(xs), num=500)
    plt.plot(x_axis_values, monoExp(x_axis_values, m, t, b), '--', label="Exponent")

    #plot GAM
    data_points_gam = gam50.generate_X_grid(term=0, n=500)
    plt.plot(data_points_gam, gam50.predict(data_points_gam), label='GAM')

    plt.legend()
    plt.axhline(y = 1, color = 'r', linestyle = '-')
    plt.xlabel("Hours since "+med_t_name+ " adminstration")
    plt.ylabel('Ratio of ' + lab_name)
    plt.savefig('plot_'+lab_name+'_'+med_t_name+'.jpeg')
    plt.clf()
    # inspect the parameters

    return(exp_params,gam50)
