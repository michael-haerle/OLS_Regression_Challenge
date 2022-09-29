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


def scale_data(train, 
               validate, 
               test, 
               cols = ['avgAnnCount', 'avgDeathsPerYear', 'incidenceRate',
       'medIncome', 'popEst2015', 'povertyPercent', 'studyPerCap',
       'MedianAge', 'MedianAgeMale', 'MedianAgeFemale',
       'AvgHouseholdSize', 'PercentMarried', 'PctNoHS18_24', 'PctHS18_24',
       'PctSomeCol18_24', 'PctBachDeg18_24', 'PctHS25_Over',
       'PctBachDeg25_Over', 'PctEmployed16_Over', 'PctUnemployed16_Over',
       'PctPrivateCoverage', 'PctPrivateCoverageAlone', 'PctEmpPrivCoverage',
       'PctPublicCoverage', 'PctPublicCoverageAlone', 'PctWhite', 'PctBlack',
       'PctAsian', 'PctOtherRace', 'PctMarriedHouseholds', 'BirthRate']):
    """
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    """
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     make the thing
    scaler = sklearn.preprocessing.MinMaxScaler()
    #     fit the thing
    scaler.fit(train[cols])
    # applying the scaler:
    train_scaled[cols] = pd.DataFrame(scaler.transform(train[cols]),
                                                  columns=train[cols].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[cols] = pd.DataFrame(scaler.transform(validate[cols]),
                                                  columns=validate[cols].columns.values).set_index([validate.index.values])
    
    test_scaled[cols] = pd.DataFrame(scaler.transform(test[cols]),
                                                 columns=test[cols].columns.values).set_index([test.index.values])
    return train_scaled, validate_scaled, test_scaled


def wrangle_df():
    """
    This function reads the data from a csv to a dataframe, fills the null values,
    creates a age bin column, and splits and scales the data.
    """
    # read csv to a dataframe
    df=pd.read_csv("cancer_reg.csv", encoding='latin-1')

    # filling nulls
    df['PctSomeCol18_24'] = 100 - (df['PctNoHS18_24'] + df['PctHS18_24'] + df['PctBachDeg18_24'])

    df['PctPrivateCoverageAlone'] =  df['PctPrivateCoverageAlone'].fillna(100 - (df['PctEmpPrivCoverage'] + df['PctPublicCoverageAlone']))

    df['AvgHouseholdSize'].where(df['AvgHouseholdSize'] >= 1, 1, inplace=True)

    # splitting the data
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

    # imputing more nulls
    x = train['PctEmployed16_Over'].mean()
    df['PctEmployed16_Over'] =  df['PctEmployed16_Over'].fillna(x)

    avg_age = round(train.MedianAge.mean())
    avg_age_male = round(train.MedianAgeMale.mean())
    avg_age_female = round(train.MedianAgeFemale.mean())
    df['MedianAgeMale'].where(df['MedianAgeMale'] <= 100, avg_age_male, inplace=True)
    df['MedianAgeFemale'].where(df['MedianAgeFemale'] <= 100, avg_age_female, inplace=True)
    df['MedianAge'].where(df['MedianAge'] <= 100, avg_age, inplace=True)

    # creating an age bin
    df['AgeBin'] = pd.qcut(df.MedianAge, 3, labels=['young', 'mid-aged', 'old'])

    # resplitting the data with the nulls filled
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

    train_scaled, validate_scaled, test_scaled = scale_data(train, validate, test)

    return df, train, validate, test, train_scaled, validate_scaled, test_scaled