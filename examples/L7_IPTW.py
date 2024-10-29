###################################################################################################################
# EPID 717: Inverse Probability of Treatment Weighting
#   Code corresponding to the first-author publication example from lecture
#
# Paul Zivich (2024/10/15)
###################################################################################################################

#############################################
# Loading Python packages used
import sys
sys.path.append("..")
import numpy as np                                        # For array management and some functions
import pandas as pd                                       # For data management
import statsmodels.api as sm                              # For statistical models (logit, KM)
import statsmodels.formula.api as smf                     # For Wilkinson-style formulas
from statsmodels.stats.weightstats import DescrStatsW     # For calculating SMD easily
import matplotlib.pyplot as plt                           # For plotting

#############################################
# Loading data from lecture

DATA_DIRECTORY="./data"
DATA_FILE_PATH=f"{DATA_DIRECTORY}/student_pub.csv"

d = pd.read_csv(DATA_FILE_PATH)
d.info()

#############################################
# Kaplan-Meier

# Step 1: stratifying data
d1 = d.loc[d['seminar'] == 1].copy()                 # Observations that attended the seminar
d0 = d.loc[d['seminar'] == 0].copy()                 # Observations that did not attend the seminar

# Step 2a: Applying Kaplan-Meier among A=1
km_n1 = sm.SurvfuncRight(d1['time'], d1['delta'])    # Fitting KM estimator with package
kmr_n1 = km_n1.summary()                             # Extracting results
risk_n1 = 1 - kmr_n1['Surv prob']                    # Getting risk from the results



# Step 2b: Applying Kaplan-Meier among A=0
km_n0 = sm.SurvfuncRight(d0['time'], d0['delta'])    # Fitting KM estimator with package
kmr_n0 = km_n0.summary()                             # Extracting results
risk_n0 = 1 - kmr_n0['Surv prob']                    # Getting risk from the results

risk_n1.info()
risk_n0.info()

# Putting results together to compute the risk difference
rdiff_n = pd.merge(risk_n1, risk_n0, how='outer',                # Outer merge risk data on time
                   left_index=True, right_index=True)            # ... on time (the index of risk_n
tzero = pd.DataFrame([[0., 0.], ],                               # Creating row of zeroes for t=0
                     columns=['Surv prob_x', 'Surv prob_y'])
rdiff_n = pd.concat([tzero, rdiff_n])                            # Adding zero row to risk difference data
rdiff_n = rdiff_n.ffill()                                        # Forward-filling all missing risks on aligned times
rdiff_n['rd'] = rdiff_n['Surv prob_x'] - rdiff_n['Surv prob_y']  # Computing the risk difference across columns

# Step 3: display results
#   Here, I am being lazy and just adding the start & end points manually, since the above function does not do so
plt.step([0., ] + list(risk_n1.index) + [6., ], [0., ] + list(risk_n1) + [list(risk_n1)[-1], ],
         where='post', color='blue', label=r'$A=1$')
plt.step([0., ] + list(risk_n0.index) + [6., ], [0., ] + list(risk_n0) + [list(risk_n0)[-1], ],
         where='post', color='red', label=r'$A=0$')
plt.ylim([-0.01, 1.01])
plt.ylabel("Risk")
plt.xlim([0, 6])
plt.xlabel("Years since seminar")
plt.title("Crude Kaplan-Meier")
plt.legend(loc=2)
plt.tight_layout()
plt.show()

#############################################
# IPTW Kaplan-Meier

# Step 1: estimate IPTW
# Step 1a: Fit a nuisance model (logistic regression)
fam = sm.families.Binomial()                                     # Specify LINK=logit, DIST=binomial
fm = smf.glm("seminar ~ age + prev_pub", d, family=fam).fit()    # GLM with Wilkinson-style formula
print(fm.summary())                                              # Displaying regression coefficient results

# Step 1b: get predicted probabilities of A=1
d['pi_a'] = fm.predict(d)                                        # Predicted probabilities of seminar attendance

# Step 1c: calculate IPTW
#   NOTE: there are a myriad of ways to program this. One version was provided in lecture. Here, I provide a few
#   different options. There may be computational or convenience reasons to prefer one method over another, but those
#   are likely to be software specific. Instead, you should recognize that there are multiple /valid/ ways to code this
# d['iptw'] = 1 / np.where(d['seminar'] == 1, d['pi_a'], 1-d['pi_a'])        # If-then statement
d['iptw'] = d['seminar'] / d['pi_a'] + (1 - d['seminar']) / (1-d['pi_a'])  # Lecture equation

# Step 2: stratifying the Data
#   While technically this step comes later, I will split the data objects here. It will make the diagnostics a bit
#   easier to code. As long as you re-run your full script each time you make a change, such a step slightly out of
#   order should not cause issues.
d1 = d.loc[d['seminar'] == 1].copy()
d0 = d.loc[d['seminar'] == 0].copy()

# Step 1d: Check weights
#   NOTE: not explored in lecture, but we can do additional calculations beyond the standardized mean differences. A
#   simple diagnostic to start with is to examine the distribution of the weights. Here, we will just look at the max,
#   min, mean, and some percentiles. Here, we want to see the mean is approximately 2, and the max isn't "too big".
print(d['iptw'].describe())
#   I get a mean of 1.98 (close to 2) and a max of 5.96 (so one observation gets to stand-in for 6). I would deem these
#   to be reasonable and move on to other diagnostics.

#   Now we will calculate the standardized mean difference from lecture. The formulas for binary W and continuous W are
#   provided in the Bonus Resources. Note that categorical variables are a bit of a pain to code (compared to binary and
#   continuous)
# # Unweighted SMD for Previous Publication
# mu1 = np.mean(d1['prev_pub'])                                                  # Mean Prev Publication among A=1
# mu0 = np.mean(d0['prev_pub'])                                                  # Mean Prev Publication among A=0
# smd_pub = (mu1 - mu0) / np.sqrt((mu1 * (1 - mu1) + mu0 * (1 - mu0)) / 2)       # SMD formula from slides
# # Unweighted SMD for Age
# mu1 = DescrStatsW(d1['age'], ddof=1)                                           # Mean,SD for age among A=1
# mu0 = DescrStatsW(d0['age'], ddof=1)                                           # Mean,SD for age among A=1
# smd_age = (mu1.mean - mu0.mean) / np.sqrt((mu1.std ** 2 + mu0.std ** 2) / 2)   # SMD formula from slides
# # Weighted SMD for Previous Publication
# mu1 = np.average(d1['prev_pub'], weights=d1['iptw'])                           # IPTW mean Prev Publication among A=1
# mu0 = np.average(d0['prev_pub'], weights=d0['iptw'])                           # IPTW mean Prev Publication among A=1
# wsmd_pub = (mu1 - mu0) / np.sqrt((mu1 * (1 - mu1) + mu0 * (1 - mu0)) / 2)      # Same formula as before
#
# mu1 = DescrStatsW(d1['age'], weights=d1['iptw'], ddof=1)                       # IPTW mean,SD for age among A=1
# mu0 = DescrStatsW(d0['age'], weights=d0['iptw'], ddof=1)                       # IPTW mean,SD for age among A=0
# wsmd_age = (mu1.mean - mu0.mean) / np.sqrt((mu1.std ** 2 + mu0.std ** 2) / 2)  # Same formula as before
# Displaying SMD results using a Love plot
# plt.figure(figsize=[6, 2])
# plt.plot(np.abs([smd_age, smd_pub]), [1.0, 2.0], 'o', color='purple', label='Unweighted', markeredgecolor='k')
# plt.plot(np.abs([wsmd_age, wsmd_pub]), [1.0, 2.0], 's', color='forestgreen', label='Weighted', markeredgecolor='k')
# plt.axvline(0.10, linestyle=':', color='gray')
# plt.yticks([1, 2], ['Age', 'PrevPub'])
# plt.ylim([0, 3])
# plt.xlim([0, 1.2])
# plt.xlabel("Absolute Standardized Mean Difference")
# plt.legend()
# plt.tight_layout()
# plt.show()

# Step 3a: weighted Kaplan-Meier for A=1
km_w1 = sm.SurvfuncRight(d1['time'], d1['delta'], freq_weights=d1['iptw'])  # Fitting weighted KM estimator with package
kmr_w1 = km_w1.summary()                                                    # Extracting results
risk_w1 = 1 - kmr_w1['Surv prob']                                           # Getting risk from results

# Step 3b: weighted Kaplan-Meier for A=0
km_w0 = sm.SurvfuncRight(d0['time'], d0['delta'], freq_weights=d0['iptw'])  # Fitting weighted KM estimator with package
kmr_w0 = km_w0.summary()                                                    # Extracting results
risk_w0 = 1 - kmr_w0['Surv prob']                                           # Getting risk from results

# Putting results together to compute the risk difference
rdiff_w = pd.merge(risk_w1, risk_w0, how='outer',                # Outer merge risk data on time
                   left_index=True, right_index=True)            # ... on time (the index of risk_w#
rdiff_w = pd.concat([tzero, rdiff_w])                            # Adding zero row to risk difference data
rdiff_w = rdiff_w.ffill()                                        # Forward-filling all missing risks on aligned times
rdiff_w['rd'] = rdiff_w['Surv prob_x'] - rdiff_w['Surv prob_y']  # Computing the risk difference across columns

# Step 4: display results
#   Here, I am being lazy and just adding the start & end points manually, since the above function does not do so
plt.step([0., ] + list(risk_w1.index) + [6., ], [0., ] + list(risk_w1) + [list(risk_w1)[-1], ],
         where='post', color='blue', linestyle=':', label=r'$A=1$')
plt.step([0., ] + list(risk_w0.index) + [6., ], [0., ] + list(risk_w0) + [list(risk_w0)[-1], ],
         where='post', color='red', linestyle=':', label=r'$A=0$')
plt.ylim([-0.01, 1.01])
plt.ylabel("Risk")
plt.xlim([0, 6])
plt.xlabel("Years since seminar")
plt.title("IPTW Kaplan-Meier")
plt.legend(loc=2)
plt.tight_layout()
plt.show()

# END
