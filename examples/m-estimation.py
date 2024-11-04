# M-estimation: introduction and applied examples
# Python (Paul Zivich 2023/06/08)

# Loading Libraries
import numpy as np                           # Numpy to manage arrays
import pandas as pd                          # Pandas for dataframes
import statsmodels.api as sm                 # Statsmodels as reference
import statsmodels.formula.api as smf        # Statsmodels R-style formulas
import scipy as sp                           # Scipy for root-finding and derivs
import delicatessen as deli                  # Delicatessen for M-estimators

# Loading Specific functions from prior libraries
from scipy.optimize import minimize, approx_fprime, newton
from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_regression
from delicatessen.utilities import inverse_logit

# Displaying verions
print("versions")
print("--------------------")
print("NumPy:       ", np.__version__)
print("SciPy:       ", sp.__version__)
print("pandas:      ", pd.__version__)
print("statsmodels: ", sm.__version__)
print("Delicatessen:", deli.__version__)

# From Table 1
d = pd.DataFrame()
d['X'] = [0, 0, 0, 0, 1, 1, 1, 1]            # X values
d['W'] = [0, 0, 1, 1, 0, 0, 1, 1]            # W values
d['Y'] = [0, 1, 0, 1, 0, 1, 0, 1]            # Y values
d['n'] = [496, 74, 113, 25, 85, 15, 15, 3]   # Counts
d['intercept'] = 1                           # Intercept term (always 1)

# Expanding rows by n
d = pd.DataFrame(np.repeat(d.values,         # Converting tabled data
                           d['n'], axis=0),  # ... by replicating counts
                 columns=d.columns)          # ... into rows for each X,W,Y
d = d[['intercept', 'X', 'W', 'Y']].copy()   # Dropping extra rows

n = d.shape[0]                               # Number of observations




# Extracting arrays for easier coding later on
X = np.asarray(d[['intercept', 'X', 'W']])   # Design matrix for regression
y = np.asarray(d['Y'])                       # Outcome in regression




def ee_logistic(theta): 
    # Estimating equation for the logistic model
    beta = np.asarray(theta)[:, None]    # Reshaping parameter array for dot product

    # Looping through each observation
    est_vals = []                        # Empty list for storage
    for i in range(n):                   # For each observation in the data
        log
        print(beta)     # ... Log-odds of Y given design
        prob_y = inverse_logit(logodds)  # ... Predicted probability of Y
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


# Solving the estimating equations for beta
theta = solve_m_estimator(stacked_equations=sum_ee,
                          init=[0, 0, 0]
                          )
print(theta)