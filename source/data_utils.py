"""Imports required Python modules."""
import pandas as pd                                       # For data management
import statsmodels.api as sm                              # For statistical models (logit, KM)


"""
zKM_estimate: Returns the KM estimate of risk of a dataset.

Parameters
data: input pandas dataframe
time: column name for time variable
delta: column name for outcome variable
weights: optional argument for weighting

Returns
risk: KM estimate of risk

"""
def zKM_estimate(data: pd.core.frame.DataFrame, time: str, delta: str, weights: None):
    km = sm.SurvfuncRight(data[time], data[delta], freq_weights = weights)   # Fitting KM estimator with package
    kmr = km.summary()                             # Extracting results
    risk = 1 - kmr['Surv prob']                    # Getting risk from the results
    return risk