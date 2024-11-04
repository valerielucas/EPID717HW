###################################################################################################################
# EPID 717: G-computation
#   Code corresponding to the first-author publication example from lecture 8
#
# Paul Zivich (2024/10/29)
###################################################################################################################

#############################################
# Loading Python packages used

import numpy as np                                        # For array management and some functions
import pandas as pd                                       # For data management
import statsmodels.api as sm                              # For statistical models (logit, KM)
import statsmodels.formula.api as smf                     # For Wilkinson-style formulas
from lifelines import CoxPHFitter                         # For Cox proportional hazards models
from lifelines import NelsonAalenFitter                   # For Nelson-Aalen to look at proportional hazards
import matplotlib.pyplot as plt                           # For plotting

#############################################
# Loading data from lecture

d = pd.read_csv("student_pub.csv")
d.info()


#############################################
# Cox Proportional Hazards Model

# Cox model with only seminar
cph = CoxPHFitter()
cph.fit(d[['time', 'delta', 'seminar']],
        duration_col='time', event_col='delta')
print("Cox model")
print(cph.summary)
print("")

# Cox model with seminar and confounders
cph = CoxPHFitter()
cph.fit(d[['time', 'delta', 'seminar', 'age', 'prev_pub']],
        duration_col='time', event_col='delta')
print("Adjusted Cox model")
print(cph.summary)
print("")

# Looking at hazards by seminar
d1 = d.loc[d['seminar'] == 1].copy()
na1 = NelsonAalenFitter(nelson_aalen_smoothing=False)
na1.fit(d1['time'], d1['delta'])
ch1 = na1.cumulative_hazard_
d0 = d.loc[d['seminar'] == 0].copy()
na0 = NelsonAalenFitter(nelson_aalen_smoothing=False)
na0.fit(d0['time'], d0['delta'])
ch0 = na0.cumulative_hazard_

# Plot of the cumulative hazards
chd = pd.merge(ch1, ch0, how='outer',                    # Outer merge cumulative hazard data on time
               left_index=True, right_index=True)        # ... on time (the index of ch)
chd = chd.ffill()                                        # Forward-filling all missing risks on aligned times
chd['hr'] = chd['NA_estimate_x'] / chd['NA_estimate_y']  # Computing the log-hazard difference across columns

# Plotting the log cumulative hazards over time
plt.step(ch0.index, np.log(ch0['NA_estimate']), where='post', color='red')
plt.step(ch1.index, np.log(ch1['NA_estimate']), where='post', color='blue')
plt.xlabel(r"$t$")
plt.ylabel(r"$\log(H(t))$")
plt.tight_layout()
# plt.savefig("../images/ph3.png", format='png', dpi=300)
plt.close()

# Plotting the log hazard ratios over time
plt.step(chd.index, np.log(chd['hr']), where='post', color='k')
plt.xlabel(r"$t$")
plt.ylabel(r"$\log(HR(t))$")
plt.tight_layout()
# plt.savefig("../images/ph4.png", format='png', dpi=300)
plt.close()

#############################################
# Marginal Structural Cox Model

# Generating IPTW
fam = sm.families.Binomial()
fm = smf.glm("seminar ~ age + prev_pub", d, family=fam).fit()
d['pi_a'] = fm.predict(d)
d['iptw'] = 1 / np.where(d['seminar'] == 1, d['pi_a'], 1-d['pi_a'])

# Weighted Cox model
cph = CoxPHFitter()
cph.fit(d[['time', 'delta', 'seminar', 'iptw']],
        duration_col='time', event_col='delta', weights_col='iptw')
print("Marginal Structural Cox model")
print(cph.summary)
print("")

#############################################
# Survival G-computation (indicator)

# Step 0: Convert to long data set
d['time_up'] = np.ceil(d['time'] * 10)                                     # Rescaling time and rounding up
max_t = np.max(d['time_up'])                                               # Finding the new maximum time
dl = pd.DataFrame(np.repeat(d.values, max_t, axis=0), columns=d.columns)   # Copying the rows the max time
dl['t_in'] = dl.groupby("id")['time_up'].cumcount()                        # Computing the time into the interval
dl['t_out'] = dl['t_in'] + 1                                               # Computing the time out of the interval
dl['event'] = np.where(dl['t_out'] == dl['time_up'], dl['delta'], 0)       # Event indicator for the time interval
dl['event'] = np.where(dl['t_out'] > dl['time_up'], np.nan, dl['event'])   # Setting intervals past last seen as NaN
dl = dl[['id', 't_out', 'event', 'seminar', 'age', 'prev_pub']].copy()     # Restricting to columns we want
print(dl)

# Step 0.b: Checking long data set conversion
print("==============================================")
print("Checking long data set conversion")
print("==============================================")
print("Expected number of rows:   ", d.shape[0] * max_t)
print("Created number of rows:    ", dl.shape[0])
print("----------------------------------------------")
print("Original time contributed: ", np.sum(d['time_up']))
print("Long time contributed:     ", dl.dropna(subset='event').shape[0])
print("----------------------------------------------")
print("Original number of events: ", np.sum(d['delta']))
print("Long number of events:     ", np.sum(dl['event']))
print("==============================================")
print("")

# Step 1: Pooled logistic model for the outcome
dl_events = dl.loc[dl['event'].notnull()].copy()                # Restricting to rows that are non-events
f = sm.families.Binomial()                                      # Binomial logistic family for GLM
plm = smf.glm("event ~ seminar*(prev_pub + age + C(t_out))",    # GLM for the specified functional form
              dl_events, family=f).fit()                        # ... without the events
print(plm.summary())
print("# of parameters:", len(plm.params))

########################
# Step 2-4: Always A=1
dl1 = dl.copy()                                       # Copy long data set
dl1['seminar'] = 1                                    # Step 2: Set A according to the plan
dl1['pr_ne'] = 1 - plm.predict(dl1)                   # Step 3.a: Generate predictions under the plan
dl1['surv'] = dl1.groupby('id')['pr_ne'].cumprod()    # Step 3.b: Get predicted survival up to time t
surv1_plf = dl1.groupby("t_out")["surv"].mean()       # Step 4: Marginalize over all the observations to get survival
risk1_plf = 1 - surv1_plf                             # Convert survival to risk


########################
# Step 2-4: Always A=0
dl0 = dl.copy()                                       # Copy long data set
dl0['seminar'] = 0                                    # Step 2: Set A according to the plan
dl0['pr_ne'] = 1 - plm.predict(dl0)                   # Step 3.a: Generate predictions under the plan
dl0['surv'] = dl0.groupby('id')['pr_ne'].cumprod()    # Step 3.b: Get predicted survival up to time t
surv0_plf = dl0.groupby("t_out")["surv"].mean()       # Step 4: Marginalize over all the observations to get survival
risk0_plf = 1 - surv0_plf                             # Convert survival to risk


#############################################
# G-computation Results
plt.step([0., ] + list(risk1_plf.index / 10) + [6., ], [0., ] + list(risk1_plf) + [list(risk1_plf)[-1], ],
         where='post', color='blue', linestyle='--', label='$A=1$')
plt.step([0., ] + list(risk0_plf.index / 10) + [6., ], [0., ] + list(risk0_plf) + [list(risk0_plf)[-1], ],
         where='post', color='red', linestyle='--', label='$A=0$')
plt.ylim([-0.01, 1.01])
plt.ylabel("Risk")
plt.xlim([0, 6])
plt.xlabel("Years since seminar")
plt.legend(loc=2)
plt.tight_layout()
# plt.savefig("../images/results_r3.png", format='png', dpi=300)
plt.show()

rd_plf = risk1_plf - risk0_plf
plt.step([0., ] + list(rd_plf.index / 10) + [6., ], [0., ] + list(rd_plf) + [list(rd_plf)[-1], ],
         where='post', color='k', linestyle='--')
plt.axhline(linestyle='--', color='gray', zorder=-1)
plt.ylim([-0.75, 0.75])
plt.ylabel(r"Risk Difference ($\psi(t)$)")
plt.xlim([0, 6])
plt.xlabel(r"Years since seminar ($t$)")
plt.tight_layout()
# plt.savefig("../images/results_rd3.png", format='png', dpi=300)
plt.show()

#############################################
# Survival G-computation (linear)

# Pooled logistic model for the outcome
dl_events = dl.loc[dl['event'].notnull()].copy()                # Restricting to rows that are non-events
f = sm.families.Binomial()                                      # Binomial logistic family for GLM
plm = smf.glm("event ~ seminar*(prev_pub + age + t_out)",       # GLM for the specified functional form
              dl_events, family=f).fit()                        # ... without the events
# print(plm.summary())

###############
# Always A=1
dl1 = dl.copy()                                     # Copy long data set
dl1['seminar'] = 1                                  # Set A according to the plan
dl1['pr_ne'] = 1 - plm.predict(dl1)                 # Generate predictions under the plan
dl1['surv'] = dl1.groupby('id')['pr_ne'].cumprod()  # Get predicted survival up to time t
surv1_pl = dl1.groupby("t_out")["surv"].mean()      # Marginalize over all the observations to get survival
risk1_pl = 1 - surv1_pl                             # Convert survival to risk

###############
# Always A=0
dl0 = dl.copy()                                     # Copy long data set
dl0['seminar'] = 0                                  # Set A according to the plan
dl0['pr_ne'] = 1 - plm.predict(dl0)                 # Generate predictions under the plan
dl0['surv'] = dl0.groupby('id')['pr_ne'].cumprod()  # Get predicted survival up to time t
surv0_pl = dl0.groupby("t_out")["surv"].mean()      # Marginalize over all the observations to get survival
risk0_pl = 1 - surv0_pl                             # Convert survival to risk


#############################################
# Comparing G-computation Results
plt.step([0., ] + list(risk1_pl.index / 10) + [6., ], [0., ] + list(risk1_pl) + [list(risk1_pl)[-1], ],
         where='post', color='blue', linestyle='-')
plt.step([0., ] + list(risk1_plf.index / 10) + [6., ], [0., ] + list(risk1_plf) + [list(risk1_plf)[-1], ],
         where='post', color='blue', linestyle='--', label='$A=1$')
plt.step([0., ] + list(risk0_pl.index / 10) + [6., ], [0., ] + list(risk0_pl) + [list(risk0_pl)[-1], ],
         where='post', color='red', linestyle='-')
plt.step([0., ] + list(risk0_plf.index / 10) + [6., ], [0., ] + list(risk0_plf) + [list(risk0_plf)[-1], ],
         where='post', color='red', linestyle='--', label='$A=0$')
plt.ylim([-0.01, 1.01])
plt.ylabel("Risk")
plt.xlim([0, 6])
plt.xlabel("Years since seminar")
plt.legend(loc=2)
plt.tight_layout()
# plt.savefig("../images/results_r5.png", format='png', dpi=300)
plt.show()

#############################################
# Survival G-computation (wider intervals)

# Step 0: Convert to long data set
d['time_up'] = np.ceil(d['time'] * 2)
max_t = np.max(d['time_up'])
dl = pd.DataFrame(np.repeat(d.values, max_t, axis=0), columns=d.columns)
dl['t_in'] = dl.groupby("id")['time_up'].cumcount()
dl['t_out'] = dl['t_in'] + 1
dl['event'] = np.where(dl['t_out'] == dl['time_up'], dl['delta'], 0)
dl['event'] = np.where(dl['t_out'] > dl['time_up'], np.nan, dl['event'])
dl = dl[['id', 't_out', 'event', 'seminar', 'age', 'prev_pub']].copy()

# Step 0.b: Checking long data set conversion
# print("==============================================")
# print("Checking long data set conversion")
# print("==============================================")
# print("Expected number of rows:   ", d.shape[0] * max_t)
# print("Created number of rows:    ", dl.shape[0])
# print("----------------------------------------------")
# print("Original time contributed: ", np.sum(d['time_up']))
# print("Long time contributed:     ", dl.dropna(subset='event').shape[0])
# print("----------------------------------------------")
# print("Original number of events: ", np.sum(d['delta']))
# print("Long number of events:     ", np.sum(dl['event']))
# print("==============================================")
# print("")

# Step 1: Pooled logistic model for the outcome
dl_events = dl.loc[dl['event'].notnull()].copy()                # Restricting to rows that are non-events
f = sm.families.Binomial()                                      # Binomial logistic family for GLM
plm = smf.glm("event ~ seminar*(prev_pub + age + C(t_out))",    # GLM for the specified functional form
              dl_events, family=f).fit()                        # ... without the events

# Step 2-4: Always A=1
dl1 = dl.copy()                                       # Copy long data set
dl1['seminar'] = 1                                    # Step 2: Set A according to the plan
dl1['pr_ne'] = 1 - plm.predict(dl1)                   # Step 3.a: Generate predictions under the plan
dl1['surv'] = dl1.groupby('id')['pr_ne'].cumprod()    # Step 3.b: Get predicted survival up to time t
surv1_plw = dl1.groupby("t_out")["surv"].mean()       # Step 4: Marginalize over all the observations to get survival
risk1_plw = 1 - surv1_plw                             # Convert survival to risk


# Step 2-4: Always A=0
dl0 = dl.copy()                                       # Copy long data set
dl0['seminar'] = 0                                    # Step 2: Set A according to the plan
dl0['pr_ne'] = 1 - plm.predict(dl0)                   # Step 3.a: Generate predictions under the plan
dl0['surv'] = dl0.groupby('id')['pr_ne'].cumprod()    # Step 3.b: Get predicted survival up to time t
surv0_plw = dl0.groupby("t_out")["surv"].mean()       # Step 4: Marginalize over all the observations to get survival
risk0_plw = 1 - surv0_plw                             # Convert survival to risk


#############################################
# Comparing G-computation Results
plt.step([0., ] + list(risk1_plw.index / 2) + [6., ], [0., ] + list(risk1_plw) + [list(risk1_plw)[-1], ],
         where='post', color='blue', linestyle='-')
plt.step([0., ] + list(risk1_plf.index / 10) + [6., ], [0., ] + list(risk1_plf) + [list(risk1_plf)[-1], ],
         where='post', color='blue', linestyle='--', label='$A=1$')
plt.step([0., ] + list(risk0_plw.index / 2) + [6., ], [0., ] + list(risk0_plw) + [list(risk0_plw)[-1], ],
         where='post', color='red', linestyle='-')
plt.step([0., ] + list(risk0_plf.index / 10) + [6., ], [0., ] + list(risk0_plf) + [list(risk0_plf)[-1], ],
         where='post', color='red', linestyle='--', label='$A=0$')
plt.ylim([-0.01, 1.01])
plt.ylabel("Risk")
plt.xlim([0, 6])
plt.xlabel("Years since seminar")
plt.legend(loc=2)
plt.tight_layout()
# plt.savefig("../images/results_r6.png", format='png', dpi=300)
plt.show()

# END
