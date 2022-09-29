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

def vis_1(train):
    """
    This function plots a scatterplot with the x as TARGET_deathRate
    ,the y as incidenceRate, and the hue as binnedInc.
    """
    # change the fig size
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='TARGET_deathRate', y='incidenceRate', data=train, hue='binnedInc')
    plt.title('There is a positive correlation with Deathrate and Incidence Rate')

def stats_1(train):
    """
    This function runs a pearsonr correlation test to determine the validity of what
    we surmised from vis_1. 
    """
    # Run the pearson correlation test and print the correlation and p value
    corr, p = stats.pearsonr(train.TARGET_deathRate, train.incidenceRate)
    print('Correlation:', corr)
    print('P-value:', p)

    # Set the alpha and print the results
    alpha = 0.05
    Null = 'There is no relationship between the Deathrate and Incidence Rate.'

    Alt = 'There is a relationship between the Deathrate and Incidence Rate.'

    if p < alpha:
        print('We reject the null hypothesis that', Null)
        print(Alt)
    else:
        print('We fail to reject the null hypothesis that', Null)

def vis_2(train):
    """
    This function plots a swarmplot with the x as binnedInc
    and the y as TARGET_deathRate.
    """
    avg_deathrate = train.TARGET_deathRate.mean()
    plt.figure(figsize=(10, 10))
    sns.swarmplot(data=train, x="binnedInc", y="TARGET_deathRate")
    plt.xticks(rotation=-70)
    plt.axhline(avg_deathrate, label='avg_deathrate', color='red')
    plt.legend()
    plt.title("The Lowest Binned Income (In red) has More Counties With a High Deathrate")

def vis_3(train):
    """
    This function plots a scatterplot with the x as TARGET_deathRate
    ,and the y as povertyPercent.
    """
    # change the fig size
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='TARGET_deathRate', y='povertyPercent', data=train)
    plt.title('Deathrate and Poverty Percent have a Positive Correlation')

def stats_3(train):
    """
    This function runs a pearsonr correlation test to determine the validity of what
    we surmised from vis_3. 
    """
    # Run the pearson correlation test and print the correlation and p value
    corr, p = stats.pearsonr(train.TARGET_deathRate, train.povertyPercent)
    print('Correlation:', corr)
    print('P-value:', p)

    # Set the alpha and print the results
    alpha = 0.05
    Null = 'There is no relationship between the Deathrate and Poverty Percent.'

    Alt = 'There is a relationship between the Deathrate and Poverty Percent.'

    if p < alpha:
        print('We reject the null hypothesis that', Null)
        print(Alt)
    else:
        print('We fail to reject the null hypothesis that', Null)

def vis_4(train):
    """
    This function plots a scatterplot with the x as TARGET_deathRate
    ,and the y as MedianAge.
    """
    # change the fig size
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='TARGET_deathRate', y='MedianAge', data=train, hue='AgeBin')
    plt.title("There isn't a Relationship Between Deathrate and Median Age")

def stats_4(train):
    """
    This function runs a pearsonr correlation test to determine the validity of what
    we surmised from vis_4. 
    """
    # Run the pearson correlation test and print the correlation and p value
    corr, p = stats.pearsonr(train.TARGET_deathRate, train.MedianAge)
    print('Correlation:', corr)
    print('P-value:', p)

    # Set the alpha and print the results
    alpha = 0.05
    Null = 'There is no relationship between the Deathrate and Median Age.'

    Alt = 'There is a relationship between the Deathrate and Median Age.'

    if p < alpha:
        print('We reject the null hypothesis that', Null)
        print(Alt)
    else:
        print('We fail to reject the null hypothesis that', Null)

def vis_5(train):
    """
    This function plots a violinplot with the x as AgeBin
    ,and the y as TARGET_deathRate.
    """
    # change the fig size
    plt.figure(figsize=(8, 8))
    # Draw a nested violinplot and split the violins for easier comparison
    sns.violinplot(data=train, x="AgeBin", y="TARGET_deathRate",
                split=True, inner="quart", linewidth=1, palette='crest')
    plt.title('Middle Aged People have a Slightly Higher Average Deathrate')

def vis_6(train):
    """
    This function plots a barplot with the x as Rate per Capita (100000)
    ,and the y as AgeBin.
    """
    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(12, 12))

    # Plot the total 
    sns.set_color_codes("pastel")
    sns.barplot(x="incidenceRate", y="AgeBin", data=train,
                label="Incident Rate", color="b")

    # Plot the 
    sns.set_color_codes("muted")
    sns.barplot(x="TARGET_deathRate", y="AgeBin", data=train,
                label="Death rate", color="b")

    # Add a legend and informative axis label
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(ylabel="",
        xlabel="Rate per capita(100_000)", title='Only 1/3 of People who were Diagnosed died from Cancer')
    sns.despine(left=True, bottom=True)

def corr_vis(train):
    """
    This function uses a relplot to mimic a heatmap. 
    This lets us see the correlations in our data.
    """
    train_corr = train.corr().stack().reset_index(name="correlation")
    g = sns.relplot(
        data=train_corr,
        x="level_0", y="level_1", hue="correlation", size="correlation",
        palette="coolwarm", hue_norm=(-.5, .5), edgecolor="1",
        height=10, sizes=(50, 250), size_norm=(-.2, .8))
    g.set(xlabel="", ylabel="", title='Deathrate has a Positive Correlation with 8 Features', aspect="equal")
    g.ax.margins(.02)
    g.map(plt.axhline, y=2, color='red', zorder=1,linewidth=18, alpha=.2)
    g.map(plt.axvline, x=2, color='red', zorder=1,linewidth=18, alpha=.2)
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)
    for artist in g.legend.legendHandles:
        artist.set_edgecolor(".7")