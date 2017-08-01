# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 10:46:40 2017

@author: whuang67
"""
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as ss
import matplotlib.pyplot as plt
# Question 1
## (a)
os.chdir("C:/Users/whuang67/downloads")
longley = pd.read_csv('longley.csv')
print(np.corrcoef(np.transpose(longley)))

## (b)
y_1, X_1b = dmatrices('Employed ~ GNP_deflator + GNP + Unemployed + Armed_Forces+ Population + Year',
                      longley,
                      return_type = 'dataframe')
X_1b = np.array(X_1b)
# model_1b = LinearRegression().fit(X_1b, y_1)
vif_1b = [variance_inflation_factor(X_1b, i) for i in range(1, X_1b.shape[1])]
print(vif_1b)

## (c)
y_1c, X_1c = dmatrices('Population ~ GNP_deflator + GNP + Unemployed + Armed_Forces + Year',
                      longley,
                      return_type = 'dataframe')
model_1c = LinearRegression(fit_intercept=False).fit(X_1c, y_1c)
print(model_1c.score(X_1c, y_1c))

## (d)
def get_resid(model, X, y):
    pred = model.predict(X)
    resid = y - pred
    return resid
_, X_1d = dmatrices('Employed ~ GNP_deflator + GNP + Unemployed + Armed_Forces + Year',
                    longley,
                    return_type = 'dataframe')
model_1d = LinearRegression(fit_intercept=False).fit(X_1d, y_1)
np.array(get_resid(model_1d, X_1d, y_1))
print(np.corrcoef(get_resid(model_1c, X_1c, y_1c).Population.values,
                  get_resid(model_1d, X_1d, y_1).Employed.values)[0][1])

## (e)
_, X_1e = dmatrices('Employed ~ Unemployed + Armed_Forces + Year',
                    longley,
                    return_type = 'dataframe')
X_1e = np.array(X_1e)
vif_1e = [variance_inflation_factor(X_1e, i) for i in range(1, X_1e.shape[1])]
print(vif_1e)

## (f)
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

model_1b = LinearRegression(fit_intercept=False).fit(X_1b, y_1)
model_1e = LinearRegression(fit_intercept=False).fit(X_1e, y_1)
print(ANOVA(model_1e, model_1b, X_1e, X_1b, y_1))

## (g)
def fitted_vs_resid(model, X, y, pointcol = "blue", linecol = "green"):
    fitted = model.predict(X)
    resid = y - fitted
    plt.scatter(fitted, resid, color = pointcol)
    plt.axhline(y = 0, color = linecol)
    plt.show()

def Normal_QQ(model, X, y):
    resid = (y - model.predict(X))
    resid = resid[resid.columns[0]]
    ss.probplot(resid, dist="norm", plot=plt)
    plt.show()

fitted_vs_resid(model_1e, X_1e, y_1)
Normal_QQ(model_1e, X_1e, y_1)



# Question 2
## (a)
odor = pd.read_csv("odor.csv", index_col = 0)
y_2, X_2a = dmatrices('odor ~ (temp+gas+pack)**2+I(temp**2)+I(gas**2)+I(pack**2)',
                      odor,
                      return_type = 'dataframe')
model_2a = LinearRegression(fit_intercept=0).fit(X_2a, y_2)

def ANOVA_1(model, X, y):
    SSR = ((model.predict(X)-y.mean().values)**2).sum()
    df_R = X.shape[1]-1
    MSR = SSR/df_R
    SSE = ((y-model.predict(X))**2).sum().values[0]
    df_E = X.shape[0] - X.shape[1]
    MSE = SSE/df_E
    F_stat = MSR/MSE
    p_val = ss.f.sf(F_stat, df_R, df_E)
    return({"p_val": p_val,
            "F_stat": F_stat,
            "df_1": df_R,
            "df_2": df_E})

print(model_2a.coef_[0])
print(ANOVA_1(model_2a, X_2a, y_2))

## (b)
_, X_2b = dmatrices('odor ~ temp+gas+pack+I(temp**2)+I(gas**2)+I(pack**2)',
                    odor,
                    return_type = 'dataframe')
model_2b = LinearRegression(fit_intercept=0).fit(X_2b, y_2)
print(model_2b.coef_[0])
print(ANOVA(model_2b, model_2a, X_2b, X_2a, y_2))

## (c)
print(model_2a.score(X_2a, y_2))
print(model_2b.score(X_2b, y_2))

## (d)
def adjusted_R_squared(model, X, y):
    r_sq = model.score(X, y)
    adj_r = r_sq-(1-r_sq)*(X.shape[1]-1.0)/(X.shape[0]-X.shape[1])
    return adj_r
print(adjusted_R_squared(model_2a, X_2a, y_2))
print(adjusted_R_squared(model_2b, X_2b, y_2))



# Question 3
## (a)
teengamb = pd.read_csv('teengamb.csv')
