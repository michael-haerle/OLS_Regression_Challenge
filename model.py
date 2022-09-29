# To get rid of those blocks of red warnings
import warnings
warnings.filterwarnings("ignore")

# Standard Imports
import numpy as np
from scipy import stats
import pandas as pd
from math import sqrt
import os
from scipy.stats import spearmanr

# Vis Imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pandas.plotting import register_matplotlib_converters


# Modeling Imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import sklearn.preprocessing
import statsmodels.api as sm

def baseline(y_train):
    """
    This function prints the RMSE baseline using the mean of TARGET_deathRate.
    """
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
        
    # 1. Predict tax_value_pred_mean
    TARGET_deathRate_pred_mean = y_train['TARGET_deathRate'].mean()
    y_train['TARGET_deathRate_pred_mean'] = TARGET_deathRate_pred_mean

    # 2. RMSE of tax_value_pred_mean
    rmse_train = mean_squared_error(y_train.TARGET_deathRate, y_train.TARGET_deathRate_pred_mean)**(1/2)

    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2))

def eval_models(y_train, y_validate, X_train, X_validate, X_test, y_test):
    """
    This function creates my top 3 models and prints their results.
    """
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)

    # create the model object
    lm = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train, y_train.TARGET_deathRate)

    # predict train
    y_train['TARGET_deathRate_pred_lm'] = lm.predict(X_train)

    # predict validate
    y_validate['TARGET_deathRate_pred_lm'] = lm.predict(X_validate)

    # Getting rid of the negative predicted value
    replace_lm = y_validate['TARGET_deathRate_pred_lm'].min()
    replace_lm_avg = y_validate['TARGET_deathRate_pred_lm'].mean()
    y_validate['TARGET_deathRate_pred_lm'] = y_validate['TARGET_deathRate_pred_lm'].replace(replace_lm, replace_lm_avg)

    # create the model object
    lars = LassoLars(alpha=1.0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lars.fit(X_train, y_train.TARGET_deathRate)

    # predict train
    y_train['TARGET_deathRate_pred_lars'] = lars.predict(X_train)

    # predict validate
    y_validate['TARGET_deathRate_pred_lars'] = lars.predict(X_validate)

    # Getting rid of the negative predicted value
    replace_lars = y_validate['TARGET_deathRate_pred_lars'].min()
    replace_lars_avg = y_validate['TARGET_deathRate_pred_lars'].mean()
    y_validate['TARGET_deathRate_pred_lars'] = y_validate['TARGET_deathRate_pred_lars'].replace(replace_lars, replace_lars_avg)

    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate)
    X_test_degree2 = pf.transform(X_test)

    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.TARGET_deathRate)

    # predict train
    y_train['TARGET_deathRate_pred_lm2'] = lm2.predict(X_train_degree2)

    # predict validate
    y_validate['TARGET_deathRate_pred_lm2'] = lm2.predict(X_validate_degree2)

    # Getting rid of the negative predicted value
    replace_lm2 = y_validate['TARGET_deathRate_pred_lm2'].min()
    replace_lm2_avg = y_validate['TARGET_deathRate_pred_lm2'].mode()
    y_validate['TARGET_deathRate_pred_lm2'] = y_validate['TARGET_deathRate_pred_lm2'].replace(replace_lm2, replace_lm2_avg[0])

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.TARGET_deathRate, y_train.TARGET_deathRate_pred_lars)**(1/2)
    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.TARGET_deathRate, y_validate.TARGET_deathRate_pred_lars)**(1/2)
    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", round(rmse_train, 2), 
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))
    print("R2 Value:", round(r2_score(y_train.TARGET_deathRate, y_train.TARGET_deathRate_pred_lars), 2))
    print('-----------------------------------------------')
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.TARGET_deathRate, y_train.TARGET_deathRate_pred_lm)**(1/2)
    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.TARGET_deathRate, y_validate.TARGET_deathRate_pred_lm)**(1/2)
    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", round(rmse_train, 2), 
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))
    print("R2 Value:", round(r2_score(y_train.TARGET_deathRate, y_train.TARGET_deathRate_pred_lm), 2))
    print('-----------------------------------------------')
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.TARGET_deathRate, y_train.TARGET_deathRate_pred_lm2)**(1/2)
    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.TARGET_deathRate, y_validate.TARGET_deathRate_pred_lm2)**(1/2)
    print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", round(rmse_train, 2), 
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))
    print("R2 Value:", round(r2_score(y_train.TARGET_deathRate, y_train.TARGET_deathRate_pred_lm2), 2))

def test_model(y_test, X_train, X_test, y_train):
    """
    This function prints the results of my best model on the test dataset.
    """
    # We need y_test to be a dataframe 
    y_test = pd.DataFrame(y_test)
    y_train = pd.DataFrame(y_train)

    # create the model object
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)
    
    # transform X_test    
    X_test_degree2 = pf.transform(X_test)
    
    # create the model object
    lm2 = LinearRegression(normalize=True)

    
    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.TARGET_deathRate)
    
    # predict test
    y_test['TARGET_deathRate_pred_lm2'] = lm2.predict(X_test_degree2)
    
    rmse_test = mean_squared_error(y_test.TARGET_deathRate, y_test.TARGET_deathRate_pred_lm2)**(1/2)
    print("RMSE for Polynomial Model, degrees=2\nTest/Out-of-Sample: ", round(rmse_test, 2))
    print("R2 Value:", round(r2_score(y_test.TARGET_deathRate, y_test.TARGET_deathRate_pred_lm2), 2))
