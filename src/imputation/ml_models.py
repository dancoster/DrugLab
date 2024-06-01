# Training ML models module
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import pandas as pd
from catboost import CatBoostClassifier
import math
#from scaling import z_score_norm_seen, z_score_norm_unseen, is_data_standardized


def NB_model(X_train, y_train, X_test, args_dict):
    nb_clf = GaussianNB(**args_dict)
    nb_clf.fit(X_train, y_train)
    prediction_proba = nb_clf.predict_proba(X_test)[:, 1]
    return prediction_proba


def RF_model(X_train, y_train, X_test, args_dict):
    rf_clf = RandomForestClassifier(**args_dict)
    rf_clf.fit(X_train, y_train)
    prediction_proba = rf_clf.predict_proba(X_test)[:, 1]
    return prediction_proba


def GBT_model(X_train, y_train, X_test, args_dict):
    gbt_clf = GradientBoostingClassifier(**args_dict)
    gbt_clf.fit(X_train, y_train)
    prediction_proba = gbt_clf.predict_proba(X_test)[:, 1]
    return prediction_proba


def LR_model(X_train, y_train, X_test, args_dict, save_coef=False):
    lr_clf = LogisticRegression(**args_dict)
    lr_clf.fit(X_train, y_train)
    prediction_proba = lr_clf.predict_proba(X_test)[:, 1]

    if save_coef:
        save_features_coef(lr_clf, X_train.columns, f"lr_{lr_clf.get_params()['penalty']}")

    return prediction_proba


def Lasso_model(X_train, y_train, X_test, standardized_features, args_dict, save_coef=False):
    # check if data was already standardized
    if not is_data_standardized(X_train, standardized_features):
        print("Perform standardization for Lasso")
        X_train, features_params = z_score_norm_seen(X_train, standardized_features)
        X_test = z_score_norm_unseen(X_test, standardized_features, features_params)

    # train and predict
    lasso_clf = Lasso(**args_dict)
    lasso_clf.fit(X_train, y_train)
    prediction = lasso_clf.predict(X_test)

    if save_coef:
        save_features_coef(lasso_clf, X_train.columns, "lasso")

    return prediction


def Ridge_model(X_train, y_train, X_test, standardized_features, args_dict, save_coef=False):
    # check if data was already standardized
    if not is_data_standardized(X_train, standardized_features):
        print("Perform standardization for Ridge")
        X_train, features_params = z_score_norm_seen(X_train, standardized_features)
        X_test = z_score_norm_unseen(X_test, standardized_features, features_params)

    # train and predict
    rr_clf = Ridge(**args_dict)
    rr_clf.fit(X_train, y_train)
    prediction = rr_clf.predict(X_test)

    # save features' coefficients if needed
    if save_coef:
        save_features_coef(rr_clf, X_train.columns, "ridge")

    return prediction


def XGB_model(X_train, y_train, X_test, args_dict):
    xgb_clf = xgb.XGBClassifier(**args_dict)
    xgb_clf.fit(X_train, y_train)
    prediction_proba = xgb_clf.predict_proba(X_test)[:, 1]
    return prediction_proba


def CatBoost_model(X_train, y_train, X_test, args_dict):
    ctb_clf = CatBoostClassifier(**args_dict)
    ctb_clf.fit(X_train, y_train, verbose=False)
    print("Catboost params: ", ctb_clf.get_params())
    prediction_proba = ctb_clf.predict_proba(X_test, verbose=False)[:, 1]
    return prediction_proba


def SVM_model(X_train, y_train, X_test, standardized_features, args_dict):
    # check if data was already standardized
    if not is_data_standardized(X_train, standardized_features):
        print(f"Perform standardization for {args_dict['kernel']} SVM")
        X_train, features_params = z_score_norm_seen(X_train, standardized_features)
        X_test = z_score_norm_unseen(X_test, standardized_features, features_params)

    if args_dict["kernel"] == 'linear':
        args = args_dict.copy()
        args.pop("kernel")
        lin_svm = svm.LinearSVC(**args)
        svm_clf = CalibratedClassifierCV(lin_svm)
    else:
        svm_clf = svm.SVC(**args_dict)  # svm Classifier

    svm_clf.fit(X_train, y_train)
    prediction_proba = svm_clf.predict_proba(X_test)[:, 1]
    return prediction_proba


def MLP_model(X_train, y_train, X_test, standardized_features, args_dict):
    if "hidden_layer_sizes" not in args_dict or args_dict["hidden_layer_sizes"] is None:
        d = len(X_train.columns)
        args_dict["hidden_layer_sizes"] = tuple([d//(2**i) for i in range(1, int(math.log(d, 2)))])

    if not is_data_standardized(X_train, standardized_features):
        print("Perform standardization for MLP")
        X_train, features_params = z_score_norm_seen(X_train, standardized_features)
        X_test = z_score_norm_unseen(X_test, standardized_features, features_params)

    mlp_clf = MLPClassifier(**args_dict)
    mlp_clf.fit(X_train, y_train)
    prediction_proba = mlp_clf.predict_proba(X_test)[:, 1]
    return prediction_proba


def save_features_coef(clf, feat_list, model_name, output_dir='../reports/'):
    """ Store features coefficients of regression models"""
    important_features = pd.DataFrame(clf.coef_).transpose()
    if model_name.find("lr"):
        important_features = important_features.transpose()
    important_features.columns = ["coef"]
    important_features["features"] = feat_list
    output_file_path = output_dir + "features_coef_by" + model_name + ".xlsx"
    important_features.sort_values(by=['coef']).to_excel(output_file_path)


def run_models(X_train, y_train, X_test, models_parameters, save_coef, lists_of_features):
    """
    A function that runs all the ml models implemented, by a dictionary containing hyperparameters for each model.
    models_parameters should be a dictionary containing the model name as the key, and a dictionary of hyperparameters
     as value
    models:
    0: NB (nb)
    1: RF (rf)
    2: GBT (gbt)
    3: LR with l2 penalty (lr_l2)
    4: LR with l1 penalty (lr_l1)
    5: LASSO (lasso)
    6: Ridge (rr)
    7: XGB (xgb)
    8: single XGB (single_xgb)
    9: linear SVM (linear_svm)
    10: rbf SVM (rbf_svm)
    11: poly SVM (poly_svm)
    12: sigmoid SVM (sigmoid_svm)
    13: CatBoost (ctb)
    14: MLP (mlp)
    """


    results = {}
    num_selected_features = [col for col in lists_of_features["features"] if
                             col not in lists_of_features["categorical"]]

    # Naive Bayes Classifier
    if "nb" in models_parameters:
        results["nb"] = NB_model(X_train, y_train, X_test, models_parameters["nb"])

    # Random Forest Classifier
    if "rf" in models_parameters:
        results["rf"] = RF_model(X_train, y_train, X_test, models_parameters["rf"])

    # Gradient Boosting trees Classifier
    if "gbt" in models_parameters:
        results["gbt"] = GBT_model(X_train, y_train, X_test, models_parameters["gbt"])

    # Logistic Regression with l2 penalty Classifier
    if "lr_l2" in models_parameters:
        results["lr_l2"] = LR_model(X_train, y_train, X_test, models_parameters["lr_l2"], save_coef)

    # Logistic Regression with l1 penalty Classifier
    if "lr_l1" in models_parameters:
        results["lr_l1"] = LR_model(X_train, y_train, X_test, models_parameters["lr_l1"], save_coef)

    # LASSO regression
    if "lasso" in models_parameters:
        results["lasso"] = Lasso_model(X_train.copy(), y_train, X_test.copy(), num_selected_features, models_parameters["lasso"], save_coef)

    # Ridge regression
    if "rr" in models_parameters:
        results["rr"] = Ridge_model(X_train.copy(), y_train, X_test.copy(), num_selected_features, models_parameters["rr"], save_coef)

    # xgboost
    if "xgb" in models_parameters:
        results["xgb"] = XGB_model(X_train, y_train, X_test, models_parameters["xgb"])

    # Single tree - xgboost
    if "single_xgb" in models_parameters:
        results["single_xgb"] = XGB_model(X_train, y_train, X_test, models_parameters["single_xgb"])

    # SVM Classifiers
    if "linear_svm" in models_parameters:
        results["linear_svm"] = SVM_model(X_train.copy(), y_train, X_test.copy(), num_selected_features, models_parameters["linear_svm"])
    if "rbf_svm" in models_parameters:
        results["rbf_svm"] = SVM_model(X_train.copy(), y_train, X_test.copy(), num_selected_features, models_parameters["rbf_svm"])
    if "poly_svm" in models_parameters:
        results["poly_svm"] = SVM_model(X_train.copy(), y_train, X_test.copy(), num_selected_features, models_parameters["poly_svm"])
    if "sigmoid_svm" in models_parameters:
        results["sigmoid_svm"] = SVM_model(X_train.copy(), y_train, X_test.copy(), num_selected_features, models_parameters["sigmoid_svm"])

    # catboost
    if "ctb" in models_parameters:
        results["ctb"] = CatBoost_model(X_train, y_train, X_test, models_parameters["ctb"])

    # MLP
    if "mlp" in models_parameters:
        results["mlp"] = MLP_model(X_train, y_train, X_test, num_selected_features, models_parameters["mlp"])


    return results