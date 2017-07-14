# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 08:39:39 2017

@author: whuang67
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from patsy import dmatrices
import scipy.stats as ss

def ANOVA(model_red, model_full, X_red, X_full, y):
    SSE_reduced = ((y-model_red.predict(X_red))**2).sum().values[0]
    SSE_full = ((y-model_full.predict(X_full))**2).sum().values[0]
    df_reduced = X_red.shape[0] - X_red.shape[1]
    df_full = X_full.shape[0] - X_full.shape[1]
    denominator = SSE_full/df_full
    numerator = (SSE_reduced-SSE_full)/(df_reduced-df_full)
    F_stat = numerator/denominator
    p_val = ss.f.sf(F_stat, df_reduced-df_full, df_full)
    return({"F_stat": F_stat,
            "df_1": df_reduced-df_full,
            "df_2": df_full,
            "p_val": p_val})

def t_test(model, X, y):
    C_diag = np.diag(np.linalg.inv(np.dot(np.transpose(X), X)))
    var = (((y-model.predict(X))**2).sum().values[0]/(X.shape[0]-X.shape[1]))
    se_betas = (var*C_diag)**.5
    t_stat = (model.coef_[0]/se_betas)[1]
    p_val = 2*ss.t.sf(abs(t_stat), X.shape[0]-X.shape[1])
    return({"t_stat": t_stat,
            "p_val": p_val})

def approximate_equal(first, second, tolerance=10**-10):
    if np.all(first - second <= tolerance):
        output = True
    elif np.all(first + second <= tolerance):
        output = True
    else:
        output = False
    return output

def ANOVA_1(model, X, y):
    SSR = ((model.predict(X)-y.mean().values)**2).sum()
    df_R = X.shape[1]-1
    MSR = SSR/df_R
    SSE = ((y-model.predict(X))**2).sum().values[0]
    df_E = X.shape[0] - X.shape[1]
    MSE = SSE/df_E
    F_stat = MSR/MSE
    p_val = ss.f.sf(F_stat, df_R, df_E)
    return p_val


# Question 1
## (a)
os.chdir("C:/Users/whuang67/downloads")
epa2015 = pd.read_csv("epa2015.csv")

## (b)
groups = epa2015.groupby(['type'])
for name, group in groups:
    plt.scatter(group.horse, group.CO2, marker='.', s=5, label = name)
plt.show()

## (c)
y, X_1c = dmatrices("CO2 ~ horse", epa2015, return_type = "dataframe")
model_1c = LinearRegression(fit_intercept=False).fit(X_1c, y)
for name, group in groups:
    plt.scatter(group.horse, group.CO2, marker='.', s=5, label = name)
plt.plot(epa2015.horse,
         epa2015.horse*model_1c.coef_[0][1] + model_1c.coef_[0][0])
plt.show()

## (d)
y, X_1d = dmatrices("CO2 ~ horse+type", epa2015, return_type = "dataframe")
model_1d = LinearRegression(fit_intercept=False).fit(X_1d, y)
coef_1d = model_1d.coef_[0]
for name, group in groups:
    plt.scatter(group.horse, group.CO2, marker='.', s=5, label = name)
plt.plot(epa2015.horse,
         epa2015.horse*coef_1d[3] + coef_1d[0])
plt.plot(epa2015.horse,
         epa2015.horse*coef_1d[3] + coef_1d[0] + coef_1d[1])
plt.plot(epa2015.horse,
         epa2015.horse*coef_1d[3] + coef_1d[0] + coef_1d[2])
plt.show()

## (e)
y, X_1e = dmatrices('CO2 ~ horse*type', epa2015, return_type = "dataframe")
model_1e = LinearRegression(fit_intercept=False).fit(X_1e, y)
coef_1e = model_1e.coef_[0]
for name, group in groups:
    plt.scatter(group.horse, group.CO2, marker='.', s=5, label = name)
plt.plot(epa2015.horse,
         epa2015.horse*coef_1e[3] + coef_1e[0])
plt.plot(epa2015.horse,
         epa2015.horse*(coef_1e[3]+coef_1e[4]) + coef_1e[0]+coef_1e[1])
plt.plot(epa2015.horse,
         epa2015.horse*(coef_1e[3]+coef_1e[5]) + coef_1e[0]+coef_1e[2])
plt.show()

## (f)
## (g)
print ANOVA(model_1c, model_1d, X_1c, X_1d, y)
## (h)
print ANOVA(model_1d, model_1e, X_1d, X_1e, y)



# Question 2
## (a)
hospital = pd.read_csv("hospital.csv")

## (b)
y_2, X_2b = dmatrices("Days ~ Charges+Pressure+Care+Race",
                      hospital,
                      return_type = "dataframe")
model_2b = LinearRegression(fit_intercept=False).fit(X_2b, y_2)

## (c)
y_2, X_2c = dmatrices("Days ~ (Charges+Pressure)*Care+Race",
                      hospital,
                      return_type = "dataframe")
model_2c = LinearRegression(fit_intercept=False).fit(X_2c, y_2)
print ANOVA(model_2b, model_2c, X_2b, X_2c, y_2)

## (d)
y_2, X_2d = dmatrices("Days ~ (Charges+Pressure)*(Care+Race)",
                      hospital,
                      return_type = "dataframe")
model_2d = LinearRegression(fit_intercept=False).fit(X_2d, y_2)
print ANOVA(model_2b, model_2d, X_2b, X_2d, y_2)

## (e)
## (f)
y_2, X_2f = dmatrices("Days ~ Charges*Pressure*Care*Race",
                      hospital,
                      return_type = "dataframe")
model_2f = LinearRegression(fit_intercept=False).fit(X_2f, y_2)
print ANOVA(model_2d, model_2f, X_2d, X_2f, y_2)



# Question 3
## (a)
fish = pd.read_csv("fish.csv")
y_3, X_3a = dmatrices("Weight ~ Length1*HeightPct*WidthPct",
                      fish,
                      return_type = "dataframe")
model_3a = LinearRegression(fit_intercept=False).fit(X_3a, y_3)

## (b)
y_3, X_3b = dmatrices("Weight ~ Length1 + HeightPct*WidthPct",
                      fish,
                      return_type = "dataframe")
model_3b = LinearRegression(fit_intercept=False).fit(X_3b, y_3)
print ANOVA(model_3b, model_3a, X_3b, X_3a, y_3)

## (c)
## (d)



# Question 4
## Preparation
n = 16
np.random.seed(671105713)
ex4 = pd.DataFrame({"groups": ['A']*(n/2) + ['B']*(n/2),
                    "values": [0]*n})
ex4['values'] = np.random.normal(10, 3, n)

num_sims = 100
lm_t = [None]*num_sims
lm_p = [None]*num_sims
tt_t = [None]*num_sims
tt_p = [None]*num_sims

## (a)
np.random.seed(671105713)
for i in range(num_sims):
    ex4['values'] = np.random.normal(10, 3, n)
    group_A = ex4['values'][ex4.groups == "A"].values
    group_B = ex4['values'][ex4.groups == "B"].values
    y_4, X_4a = dmatrices("values~groups", ex4, return_type="dataframe")
    model_4a = LinearRegression(fit_intercept=False).fit(X_4a, y_4)
    lm_result = t_test(model_4a, X_4a, y_4)
    t_result = ss.ttest_ind(group_A, group_B, equal_var=False)
    lm_t[i] = lm_result['t_stat']
    lm_p[i] = lm_result['p_val']
    tt_t[i] = t_result.statistic
    tt_p[i] = t_result.pvalue

## (b)
print np.mean(np.array(lm_t) == np.array(tt_t))

## (c)
print np.mean(np.array(lm_p) == np.array(tt_p))

## (d)
print approximate_equal(np.array(lm_t), np.array(tt_t))
print approximate_equal(np.array(lm_p), np.array(tt_p))



# Question 5
## (a)
mean_A = 60
mean_B = 63
mean_C = 65
common_sd = 4
group_size = 7

np.random.seed(1)
ex5 = pd.DataFrame(
        {"treat": ["A"]*group_size + ["B"]*group_size + ["C"]*group_size,
         "wpm": np.random.normal(mean_A, common_sd, group_size).tolist() + \
                np.random.normal(mean_B, common_sd, group_size).tolist() + \
                np.random.normal(mean_C, common_sd, group_size).tolist()})

y_5, X_5a = dmatrices("wpm ~ treat", ex5, return_type="dataframe")
model_5a = LinearRegression(fit_intercept=False).fit(X_5a, y_5)
coef_5a = model_5a.coef_[0]
print "mu_A is {}".format(coef_5a[0])
print "mu_B is {}".format(coef_5a[0]+coef_5a[1])
print "mu_C is {}".format(coef_5a[0]+coef_5a[2])

## (b)
num_sims = 1000
p_vals = [None]*num_sims
np.random.seed(1)
for i in range(1000):
    wpm = np.random.normal(mean_A, common_sd, group_size).tolist() + \
          np.random.normal(mean_B, common_sd, group_size).tolist() + \
          np.random.normal(mean_C, common_sd, group_size).tolist()
    ex5['wpm'] = np.array(wpm)
    y_5, X_5b = dmatrices("wpm ~ treat", ex5, return_type="dataframe")
    model_5b = LinearRegression(fit_intercept=False).fit(X_5b, y_5)
    p_vals[i] = ANOVA_1(model_5b, X_5b, y_5)
# print p_vals
plt.hist(p_vals, 40)
plt.show()

## (c)
print np.mean(np.array(p_vals) < 0.1)
print np.mean(np.array(p_vals) < 0.05)
print np.mean(np.array(p_vals) < 0.01)