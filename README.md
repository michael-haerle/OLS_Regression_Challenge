# <a name="top"></a>OLS Regression Challenge
![]()

by: Michael Haerle

<p>
  <a href="https://github.com/michael-haerle" target="_blank">
    <img alt="Michael" src="https://img.shields.io/github/followers/michael-haerle?label=Follow_Michael&style=social" />
  </a>
</p>


***
[[Project Description](#project_description)]
[[Project Planning](#planning)]
[[Key Findings](#findings)]
[[Data Dictionary](#dictionary)]
[[Data Acquire and Prep](#wrangle)]
[[Data Exploration](#explore)]
[[Statistical Analysis](#stats)]
[[Modeling](#model)]
[[Conclusion](#conclusion)]
___


## <a name="project_description"></a>Project Description:
Using the data science pipeline to practice with regression. In this repository you will find everything you need to replicate this project. This project is a OLS regression challenge that finds key features of lung cancer mortality rate and uses them to predict it with the least amount of error.

[[Back to top](#top)]

***
## <a name="planning"></a>Project Planning: 
[[Back to top](#top)]

### Project Outline:
- Create README.md with data dictionary, project goals, come up with questions to lead the exploration and the steps to reproduce.
- Acquire data from data.world and automate it in a function, and store the function in the wrangle.py module.
- Clean and prepare data for exploration. Create a function to automate the process, store the function in the wrangle.py module, and prepare data in Final Report Notebook by importing and using the funtion.
- Produce at least 6 clean and easy to understand visuals.
- Clearly define hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, and document findings and takeaways.
- Scale the data for modeling.
- Establish a baseline accuracy.
- Train three different regression models.
- Evaluate models on train and validate datasets.
- Choose the model with that performs the best and evaluate that single model on the test dataset.
- Document conclusions, takeaways, and next steps in the Final Report Notebook.


### Project goals: 
- My goal is to find key features of lung cancer mortality rate and use them to predict it with the least amount of error.


### Target variable:
- The target variable for this project is deathrate.

### Initial questions:
- Where are the correlations in the data?
- Is there a relationship between Deathrate and Incidence Rate?
- Is Deathrate and Binned Income related?
- Is there a relationship between Deathrate and Poverty Percent?
- Is there a relationship between Deathrate and Median Age?
- Does a specific age bin have affect the deathrate?
- How many people that died from cancer were diagnosed?

### Need to haves (Deliverables):
- A final report notebook
- A README


### Nice to haves (With more time):
- If I had more time with the data I would implement clustering to see if I can impove the models performance.
- I would also pull more of the raw data to see if I could feature engineer more columns.


### Steps to Reproduce:
- Download the csv from https://data.world/nrippner/ols-regression-challenge , download the wrangle.py, explore.py,  model.py, and final_report.ipynb
- Make sure these are all in the same directory and run the final_report.ipynb.

***

## <a name="findings"></a>Key Findings:
[[Back to top](#top)]

- Only 1/3 of the people diagnosed actually died.
- The Mid-Aged bin has a slightly higher average deathrate then young or old does.
- The red binned [22640, 34218.1] income has more counties with extreme levels of deathrate.
- The blue binned (61494.5, 125635] income has a majority of their counties below the average deathrate. 


***

## <a name="dictionary"></a>Data Dictionary  
[[Back to top](#top)]

### Data Used

- (a): years 2010-2016
- (b): 2013 Census Estimates
---
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
| TARGET_deathRate| Dependent variable. Mean per capita (100,000) cancer mortalities(a) | float64 |
| avgAnnCount| Mean number of reported cases of cancer diagnosed annually(a) | float64 |
| avgDeathsPerYear| Mean number of reported mortalities due to cancer(a) | int64 |
| incidenceRate| Mean per capita (100,000) cancer diagoses(a) | float64 |
| medianIncome| Median income per county (b) | int64 |
| popEst2015| Population of county (b) | int64 |
| povertyPercent| Percent of populace in poverty (b) | float64 |
| studyPerCap| Per capita number of cancer-related clinical trials per county (a) | float64 |
| binnedInc| Median income per capita binned by decile (b) | object |
| MedianAge| Median age of county residents (b) | float64 |
| MedianAgeMale| Median age of male county residents (b) | float64 |
| MedianAgeFemale| Median age of female county residents (b) | float64 |
| Geography| County name (b) | object |
| AvgHouseholdSize| Mean household size of county (b) | float64 |
| PercentMarried| Percent of county residents who are married (b) | float64 |
| PctNoHS18_24| Percent of county residents ages 18-24 highest education attained: less than high school (b) | float64 |
| PctHS18_24| Percent of county residents ages 18-24 highest education attained: high school diploma (b) | float64 |
| PctSomeCol18_24| Percent of county residents ages 18-24 highest education attained: some college (b) | float64 |
| PctBachDeg18_24| Percent of county residents ages 18-24 highest education attained: bachelor's degree (b) | float64 |
| PctHS25_Over| Percent of county residents ages 25 and over highest education attained: high school diploma (b) | float64 |
| PctBachDeg25_Over| Percent of county residents ages 25 and over highest education attained: bachelor's degree (b) | float64 |
| PctEmployed16_Over| Percent of county residents ages 16 and over employed (b) | float64 |
| PctUnemployed16_Over| Percent of county residents ages 16 and over unemployed (b) | float64 |
| PctPrivateCoverage: Percent of county residents with private health coverage (b) | float64 |
| PctPrivateCoverageAlone| Percent of county residents with private health coverage alone (no public assistance) (b) | float64 |
| PctEmpPrivCoverage| Percent of county residents with employee-provided private health coverage (b) | float64 |
| PctPublicCoverage| Percent of county residents with government-provided health coverage (b) | float64 |
| PctPubliceCoverageAlone| Percent of county residents with government-provided health coverage alone (b) | float64 |
| PctWhite| Percent of county residents who identify as White (b) | float64 |
| PctBlack| Percent of county residents who identify as Black (b) | float64 |
| PctAsian| Percent of county residents who identify as Asian (b) | float64 |
| PctOtherRace| Percent of county residents who identify in a category which is not White, Black, or Asian (b) | float64 |
| PctMarriedHouseholds| Percent of married households (b) | float64 |
| BirthRate| Number of live births relative to number of women in county (b) | float64 |
| AgeBin| MedianAge binned into 3 parts, young, mid-aged, old | category |
***

## <a name="wrangle"></a>Data Acquisition and Preparation
[[Back to top](#top)]

![]()


### Prepare steps: 
- Dropped columns not needed
- Removed ouliers
- Imputed nulls with mean for PctEmployed16_Over
- Replaced outliers for MedianAge, MedianAgeMale, MedianAgeFemale that were over 100 years
- Used MedianAge to make a new column with 3 age bins
- Split into the train, validate, and test sets
- Scaled the data to be used later in modeling

*********************

## <a name="explore"></a>Data Exploration:
[[Back to top](#top)]
- Python files used for exploration:
    - wrangle.py
    - explore.py
    - model.py


### Takeaways from exploration:
- Only 1/3 of the people diagnosed actually died.
- The Mid-Aged bin has a slightly higher average deathrate then young or old does.
- The red binned [22640, 34218.1] income has more counties with extreme levels of deathrate.
- The blue binned (61494.5, 125635] income has a majority of their counties below the average deathrate.
***

## <a name="stats"></a>Statistical Analysis
[[Back to top](#top)]

### Stats Test 1: PearsonR


#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is: There is no relationship between deathrate and incidence rate.
- The alternate hypothesis (H<sub>1</sub>) is: There is a relationship between deathrate and incidence rate.

#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:
- Correlation: 0.44
- P-value: 6.73e-83
- We reject the null hypothesis that There is no relationship between the Deathrate and Incidence Rate.
- There is a relationship between the Deathrate and Incidence Rate.


### Stats Test 2: PearsonR


#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is: There is no relationship between deathrate and poverty percent.
- The alternate hypothesis (H<sub>1</sub>) is: There is a relationship between deathrate and poverty percent.

#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05


#### Results:
- Correlation: 0.42
- P-value: 1.14e-72
- We reject the null hypothesis that There is no relationship between the Deathrate and Poverty Percent.
- There is a relationship between the Deathrate and Poverty Percent.


### Stats Test 3: PearsonR


#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is: There is no relationship between deathrate and median age.
- The alternate hypothesis (H<sub>1</sub>) is: There is a relationship between deathrate and median age.

#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05


#### Results:
- Correlation: -0.0066
- P-value: 0.78
- We fail to reject the null hypothesis that There is no relationship between the Deathrate and Median Age.


***

## <a name="model"></a>Modeling:
[[Back to top](#top)]

### Baseline (Using Mean)
    
- Baseline RMSE: 27.83
    

- Selected features to input into models:
    - features =  ['incidenceRate', 'povertyPercent', 'PctHS25_Over', 'PctBachDeg25_Over', 
    'PctPublicCoverageAlone', 'PctHS18_24', 'PctUnemployed16_Over', 'PctPublicCoverage']
    
***

## Models:


### Model 1: Lasso + Lars


Model 1 results:
- RMSE for Lasso + Lars
- Training/In-Sample:  27.83 
- Validation/Out-of-Sample:  27.42
- R2 Value: 0.0


### Model 2 : OLS using LinearRegression


Model 2 results:
- RMSE for OLS using LinearRegression
- Training/In-Sample:  20.25 
- Validation/Out-of-Sample:  18.96
- R2 Value: 0.47


### Model 3 : Polynomial Model

Model 3 results:
- RMSE for Polynomial Model, degrees=2
- Training/In-Sample:  19.11 
- Validation/Out-of-Sample:  19.14
- R2 Value: 0.53


## Selecting the Best Model:

### Use Table below as a template for all Modeling results for easy comparison:

| Model | Validation | R2 |
| ---- | ---- | ---- |
| Baseline | 27.83 | 0.0 |
| Lasso + Lars | 27.42 | 0.0 |
| OLS using LinearRegression | 18.96 |  0.47 |
| Polynomial Model | 19.14 | 0.53 |


- {Polynomial Model} model performed the best


## Testing the Model

- Model Testing Results: RMSE 20.49, R2 0.46

***

## <a name="conclusion"></a>Conclusion:

- Only 1/3 of the people diagnosed actually died.
- The Mid-Aged bin has a slightly higher average deathrate then young or old does.
- The red binned [22640, 34218.1] income has more counties with extreme levels of deathrate.
- The blue binned (61494.5, 125635] income has a majority of their counties below the average deathrate.
- Our RMSE value for our test dataset beat our baseline by 26%.

#### In order to potentially reduce the deathrate in certain counties, we need to find a way to incentivise people in those lower income ranges to get checked more often. This can possibly be done through a public program of some sort. We can also spread more awareness to get checked in the first place.

[[Back to top](#top)]
