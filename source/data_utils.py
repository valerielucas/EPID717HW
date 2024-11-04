"""Imports required Python modules."""
import pandas as pd                                       # For data management
import statsmodels.api as sm                              # For statistical models (logit, KM)
import numpy as np                                       # For data management


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
def zKM_estimate(data: pd.core.frame.DataFrame, time: str, delta: str, weights = None):
    km = sm.SurvfuncRight(data[time], data[delta], freq_weights = weights)   # Fitting KM estimator with package
    kmr = km.summary()                             # Extracting results
    risk = 1 - kmr['Surv prob']                    # Getting risk from the results
    return risk



"""
ee_logistic

"""
def ee_logistic(theta):
    # Estimating equation for the logistic model
    beta = np.asarray(theta)[:, None]    # Reshaping parameter array for dot product

    # Looping through each observation
    est_vals = []                        # Empty list for storage
    for i in range(n):                   # For each observation in the data
        logodds = np.dot(X[i], beta)     # ... Log-odds of Y given design
        prob_y = np.inverse_logit(logodds)  # ... Predicted probability of Y
        v_i = (y[i] - prob_y)*X[i]       # ... Estimating function for O_i
        est_vals.append(v_i)             # ... Storing contribution
    
    # Return estimating functions stacked together
    return np.asarray(est_vals).T


def sum_ee(theta):
    # Function to sum the previous estimating equation over all i's
    stacked_equations = np.asarray(ee_logistic(theta))  # Returning stacked equation
    vals = ()                                           # Create empty tuple
    for i in stacked_equations:                         # Go through each individual theta
        vals += (np.sum(i), )                           # Add the theta sum to the tuple of thetas

    # Return the calculated values of theta
    return vals


def solve_m_estimator(stacked_equations, init):
    # Wrapper function for SciPy root-finding 
    psi = newton(stacked_equations,    # stacked equations to solve (written as sums)
                 x0=np.asarray(init),  # initial values for solver
                 maxiter=2000,         # Increasing iterations
                 disp=True)            # Raise RuntimeError if doesn't converge
    return psi



def psi(theta):
    # Dividing parameters into corresponding parts and labels from slides
    alpha = theta[0:2]                    # Logistic model coefficients
    mu0, mu1 = theta[2], theta[3]         # Causal risks
    delta1 = theta[4]                     # Causal contrast

    # Using built-in regression model functionality from delicatessen
    ee_logit = ee_regression(theta=alpha,             # Regression model
                             y=d['X'],                # ... for exposure
                             X=d[['intercept', 'W']], # ... given confounders
                             model='logistic')        # ... logistic model

    # Transforming logistic model coefficients into causal parameters
    pscore = inverse_logit(np.dot(d[['intercept', 'W']], alpha))  # Propensity score
    wt = d['X']/pscore + (1-d['X'])/(1-pscore)                    # Corresponding weights

    # Estimating function for causal risk under a=1
    ee_r1 = d['X']*d['Y']*wt - mu1                   # Weighted conditional mean
    
    # Estimating function for causal risk under a=0
    ee_r0 = (1-d['X'])*d['Y']*wt - mu0               # Weighted conditional mean
    
    # Estimating function for causal risk difference
    ee_rd = np.ones(d.shape[0])*((mu1 - mu0) - delta1)

    # Returning stacked estimating functions in order of parameters
    return np.vstack([ee_logit,   # EF of logistic model
                      ee_r0,      # EF of causal risk a=0
                      ee_r1,      # EF of causal risk a=1
                      ee_rd])     # EF of causal contrast