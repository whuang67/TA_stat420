# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 10:46:40 2017

@author: Wenke Huang
"""

import os
import pandas as pd
import numpy as np
import math
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
odor = pd.read_csv("odor.csv")
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
model_2b = LinearRegression(fit_intercept=False).fit(X_2b, y_2)
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
def get_AIC(model, X, y):
    n = X.shape[0]
    p = X.shape[1]
    pred = model.predict(X)
    RSS = ((y - pred)**2).sum()[0]
    AIC = n*math.log(RSS/n) + 2*p
    return AIC
def get_BIC(model, X, y):
    n = X.shape[0]
    p = X.shape[1]
    pred = model.predict(X)
    RSS = ((y - pred)**2).sum()[0]
    BIC = n*math.log(RSS/n) + math.log(n)*p
    return BIC

def backward(data, response, use="AIC", trace=True):
    columns = data.columns.tolist()
    columns.remove(response)
    metric = {}
    y_, X_ = data[response].to_frame(name = response), data.drop(response, axis=1)
    X_['Intercept'] = [1]*X_.shape[0]
    model = LinearRegression(fit_intercept=False).fit(X_, y_)
    if use == "AIC":
        metric['full'] = get_AIC(model, X_, y_)
    elif use == "BIC":
        metric['full'] = get_BIC(model, X_, y_)
    for var in columns:
        full_name = ""
        for e in columns:
            if e != var:
                if full_name != "":
                    full_name = full_name+"+"+e
                else:
                    full_name = response+str('~')+e
        y_, X_ = dmatrices(full_name, data, return_type="dataframe")
        model = LinearRegression(fit_intercept=False).fit(X_, y_)
        if use == "AIC":
            metric[var] = get_AIC(model, X_, y_)
        elif use == "BIC":
            metric[var] = get_BIC(model, X_, y_)
    
    mini = metric['full']
    mini_key = 'full'
    for key, val in metric.items():
        if val < mini:
            mini = val
            mini_key = key
    
    if mini_key == "full":
        kept_vars = data.drop(response, axis=1).columns.tolist()
        if trace == True:
            print("Variables in the final model: {}".format(kept_vars))
        return kept_vars
    else:
        if trace == True:
            print("Remove variable: {} ({}).".format(mini_key, mini))
        dat = data.drop(mini_key, axis = 1)
        return backward(dat, response, use=use, trace=trace)
        
print(backward(teengamb, "gamble"))
print(backward(teengamb, "gamble", use = "BIC"))

## (c)
y_3, X_3a = dmatrices("gamble ~ sex+income+verbal",
                      teengamb, return_type = "dataframe")
model_3a = LinearRegression(fit_intercept=False).fit(X_3a, y_3)
_, X_3b = dmatrices("gamble ~ sex+income",
                    teengamb, return_type = "dataframe")
model_3b = LinearRegression(fit_intercept=False).fit(X_3b, y_3)
print(ANOVA(model_3b, model_3a, X_3b, X_3a, y_3))

## (d)  ######################################################### Not finished
##############################################################################
##############################################################################
_, X_3d = dmatrices("gamble ~ sex * status * income * verbal",
                    teengamb, return_type="dataframe")
del X_3d["Intercept"]
X_3d['gamble'] = teengamb.gamble.tolist()

## (e)
print(adjusted_R_squared(model_3a, X_3a, y_3))
print(adjusted_R_squared(model_3b, X_3b, y_3))



# Question 4

# Question 5

# Question 6
## (a)
beta_0, beta_1,beta_2,beta_3,beta_4,beta_5,beta_6,beta_7,beta_8,beta_9,beta_10,sigma = \
    1,1,1,1,1,0,0,0,0,0,0,2
signif = ["Intercept", "x_1", "x_2", "x_3", "x_4"]
not_sig = ["x_5", "x_6", "x_7", "x_8", "x_9", "x_10"]
np.random.seed(42)
dat_dict = {"x_1": np.random.uniform(0, 10, 100).tolist(),
            "x_2": np.random.uniform(0, 10, 100).tolist(),
            "x_3": np.random.uniform(0, 10, 100).tolist(),
            "x_4": np.random.uniform(0, 10, 100).tolist(),
            "x_5": np.random.uniform(0, 10, 100).tolist(),
            "x_6": np.random.uniform(0, 10, 100).tolist(),
            "x_7": np.random.uniform(0, 10, 100).tolist(),
            "x_8": np.random.uniform(0, 10, 100).tolist(),
            "x_9": np.random.uniform(0, 10, 100).tolist(),
            "x_10": np.random.uniform(0, 10, 100).tolist()}
dat = pd.DataFrame(dat_dict)
dat["y"] = beta_0 + beta_1*dat["x_1"] + beta_2*dat["x_2"] + \
           beta_3*dat["x_3"] + beta_4*dat["x_4"] + \
           np.random.normal(0, sigma, 100)

np.random.seed(42)
fn_aic, fp_aic, fn_bic, fp_bic = [], [], [], []
for i in range(50):
    print(i)
    dat["y"] = beta_0 + beta_1*dat["x_1"] + beta_2*dat["x_2"] + \
               beta_3*dat["x_3"] + beta_4*dat["x_4"] + \
               np.random.normal(0, sigma, 100)
    output = backward(dat, "y", trace=False)
    output.append("Intercept")
    fn_aic.append(5-sum([e in signif for e in output]))
    fp_aic.append(sum([f in output for f in not_sig]))
    output_2 = backward(dat, "y", use="BIC", trace=False)
    output_2.append("Intercept")
    fn_bic.append(5-sum([e2 in signif for e2 in output_2]))
    fp_bic.append(sum([f2 in output_2 for f2 in not_sig]))

print(np.mean(fn_aic))  
print(np.mean(fp_aic))
print(np.mean(fn_bic))
print(np.mean(fp_bic))
    
## (b)
np.random.seed(42)
dat_dict = {"x_1": np.random.uniform(0, 10, 100).tolist(),
            "x_2": np.random.uniform(0, 10, 100).tolist(),
            "x_3": np.random.uniform(0, 10, 100).tolist(),
            "x_4": np.random.uniform(0, 10, 100).tolist(),
            "x_5": np.random.uniform(0, 10, 100).tolist(),
            "x_6": np.random.uniform(0, 10, 100).tolist(),
            "x_7": np.random.uniform(0, 10, 100).tolist()}
dat = pd.DataFrame(dat_dict)
dat['x_8'] = dat.x_1 + np.random.normal(0, 0.1, 100)
dat['x_9'] = dat.x_1 + np.random.normal(0, 0.1, 100)
dat['x_10'] = dat.x_2 + np.random.normal(0, 0.1, 100)

fn_aic, fp_aic, fn_bic, fp_bic = [], [], [], []
for i in range(50):
    print(i)
    dat["y"] = beta_0 + beta_1*dat["x_1"] + beta_2*dat["x_2"] + \
               beta_3*dat["x_3"] + beta_4*dat["x_4"] + \
               np.random.normal(0, sigma, 100)
    output = backward(dat, "y", trace=False)
    output.append("Intercept")
    fn_aic.append(5-sum([e in signif for e in output]))
    fp_aic.append(sum([f in output for f in not_sig]))
    output_2 = backward(dat, "y", use="BIC", trace=False)
    output_2.append("Intercept")
    fn_bic.append(5-sum([e2 in signif for e2 in output_2]))
    fp_bic.append(sum([f2 in output_2 for f2 in not_sig]))

print(np.mean(fn_aic))  
print(np.mean(fp_aic))
print(np.mean(fn_bic))
print(np.mean(fp_bic))
