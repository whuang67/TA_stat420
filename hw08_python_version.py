# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:55:27 2017

@author: Wenke Huang
"""

import os
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from patsy import dmatrices
import scipy.stats as ss
import matplotlib.pyplot as plt


# Question 1
## (a)
np.random.seed(420)
x0 = np.array([1]*50)
x1 = np.random.uniform(-1, 1, 50)
x2 = np.random.uniform(-1, 1, 50)
x1_2 = x1**2
x2_2 = x2**2
x1_x2 = x1*x2
y = 6+5*x1+4*x2+3*x1_2+2*x2_2+x1_x2+np.random.normal(0, 2, 50)
dat_dict = {'x0': x0.tolist(),
            'x1': x1.tolist(),
            'x2': x2.tolist(),
            'x1_2': x1_2.tolist(),
            'x2_2': x2_2.tolist(),
            'x1_x2': x1_x2.tolist()}
dat = pd.DataFrame(dat_dict)
def get_betas(X, y):
    output = np.matmul(
            np.matmul(
                    np.linalg.inv(np.matmul(np.transpose(X),
                                          X)), np.transpose(X)), y)
    return output.tolist()
np.random.seed(420)
beta0, beta1, beta2, beta12, beta22, beta1_2 = [], [], [], [], [], []
for i in range(1000):
    y = 6+5*x1+4*x2+3*x1_2+2*x2_2+x1_x2+np.random.normal(0, 2, 50)
    beta_s = get_betas(dat, y)
    beta0.append(beta_s[0])
    beta1.append(beta_s[1])
    beta2.append(beta_s[2])
    beta12.append(beta_s[3])
    beta22.append(beta_s[4])
    beta1_2.append(beta_s[5])
    
plt.hist(beta0, 20)
plt.title("Histogram of Beta_0")
plt.show()
plt.hist(beta1, 20)
plt.title("Histogram of Beta_1")
plt.show()
plt.hist(beta2, 20)
plt.title("Histogram of Beta_2")
plt.show()
plt.hist(beta12, 20)
plt.title("Histogram of Beta_12")
plt.show()
plt.hist(beta22, 20)
plt.title("Histogram of Beta_22")
plt.show()
plt.hist(beta1_2, 20)
plt.title("Histogram of Beta_1_2")
plt.show()



# Question 2
## (a)
np.random.seed(420)
dat_2a = pd.DataFrame({'x1': np.random.uniform(0, 1, 30), 
                       'x2': np.random.uniform(0, 1, 30),
                       'y_2a': 1+x1+x2+np.random.normal(0, .01, 30)})
y_2a, X_2a = dmatrices("y_2a ~ x1+x2",
                       dat_2a, return_type="dataframe")
model_2a = LinearRegression(fit_intercept=False).fit(X_2a, y_2a)
print(model_2a.score(X_2a, y_2a))

## (b)
np.random.seed(420)
dat_2b = pd.DataFrame({'x1': np.random.uniform(0, 1, 30), 
                       'x2': np.random.uniform(0, 1, 30),
                       'y_2b': 1+x2+np.random.normal(0, 1, 30)})
y_2b, X_2b = dmatrices("y_2b ~ x1+x2",
                       dat_2b, return_type = "dataframe")
model_2b = LinearRegression(fit_intercept=False).fit(X_2b, y_2b)
print(model_2b.score(X_2b, y_2b))

## (c)
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
print(ANOVA_1(model_2a, X_2a, y_2a))

## (d)
np.random.seed(420)
dat_2d = pd.DataFrame({'x1': np.random.uniform(0, 1, 30), 
                       'x2': np.random.uniform(0, 1, 30),
                       'y_2d': 1+x1+50*x2+np.random.normal(0, 1, 30)})
y_2d, X_2d = dmatrices("y_2d ~ x1+x2",
                       dat_2d, return_type = "dataframe")
model_2d = LinearRegression(fit_intercept=False).fit(X_2d, y_2d)
print(model_2d.score(X_2d, y_2d))
print(ANOVA_1(model_2d, X_2d, y_2d))

## (e)
np.random.seed(420)
dat_2e = pd.DataFrame({'x1': np.random.uniform(0, 1, 30), 
                       'x2': np.random.uniform(0, 1, 30),
                       'y_2e': 1+x1+x2+np.random.normal(0, 10, 30)})
y_2e, X_2e = dmatrices("y_2e ~ x1+x2",
                       dat_2e, return_type = "dataframe")
model_2e = LinearRegression(fit_intercept=False).fit(X_2e, y_2e)
print(model_2e.score(X_2e, y_2e))
print(ANOVA_1(model_2e, X_2e, y_2e))



# Question 3
os.chdir("C:/Users/whuang67/downloads")
two = pd.read_csv('two_way_data.csv')

cement_ = [a for a in two.cement.unique()]
curing_ = [b for b in two.curing.unique()]
for_predicting = np.array([[c, d] for c in cement_ for d in curing_])
for_predicting = pd.DataFrame(for_predicting, columns = ['cement', 'curing'])
## (a)
y_3, X_3a = dmatrices("strength ~ 1",
                      two, return_type="dataframe")
model_3a = LinearRegression(fit_intercept=False).fit(X_3a, y_3)

## (b)
_, X_3b = dmatrices("strength ~ cement",
                    two, return_type = "dataframe")
model_3b = LinearRegression(fit_intercept=False).fit(X_3b, y_3)

## (c)
_, X_3c = dmatrices("strength ~ curing",
                    two, return_type = "dataframe")
model_3c = LinearRegression(fit_intercept=False).fit(X_3c, y_3)

## (d)
_, X_3d = dmatrices("strength ~ cement+curing",
                    two, return_type = "dataframe")
model_3d = LinearRegression(fit_intercept=False).fit(X_3d, y_3)

## (e)
_, X_3e = dmatrices("strength ~ cement*curing",
                    two, return_type = "dataframe")
model_3e = LinearRegression(fit_intercept=False).fit(X_3e, y_3)

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
print(ANOVA(model_3d, model_3e, X_3d, X_3e, y_3))



# Question 4
## (a)
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

body = pd.read_csv("body.csv")
_, X = dmatrices("1 ~ s1+s2+s3+s4+s5+s6+s7+s8+s9+Age+Height+Gender+Weight",
                 body, return_type="dataframe")
del X["Intercept"]
output = backward(X, "Weight")
for i in range(len(output)):
    if i == 0:
        string = "Weight~"+output[i]
    else:
        string = string+'+'+output[i]
y_4, X_4a = dmatrices(string, body, return_type='dataframe')
model_4a = LinearRegression(fit_intercept=False).fit(X_4a, y_4)
print(model_4a.coef_)

## (b)
_, X = dmatrices("1 ~ g1+g2+g3+g4+g5+g6+g7+g8+g9+g10+g11+g12+Age+Height+Gender+Weight",
                 body, return_type="dataframe")
del X["Intercept"]
output = backward(X, "Weight")
for i in range(len(output)):
    if i == 0:
        string = "Weight~"+output[i]
    else:
        string = string+'+'+output[i]
_, X_4b = dmatrices(string, body, return_type="dataframe")
model_4b = LinearRegression(fit_intercept=False).fit(X_4b, y_4)
print(model_4b.coef_)

## (c)
def get_Leverage(X):
    H = np.matmul(np.matmul(X, np.linalg.inv(np.matmul(np.transpose(X), X))),
                  np.transpose(X))
    h_s = np.diag(H)
    return(h_s)

def get_loocv_rmse(model, X, y):
    resid = y - model.predict(X)
    output = (((np.transpose(resid.values)[0]/(1-get_Leverage(X)))**2).mean())**.5
    return output

print(get_loocv_rmse(model_4a, X_4a, y_4))


## (d)
############################## Not finished #################################
#############################################################################
#############################################################################

# Question 5
## (a)
ballbearings = pd.read_csv("ballbearings.csv")
y_5, X_5a = dmatrices(
        "np.log(L50)~ np.log(P)+np.log(Z)+np.log(D)+Company+Type",
        ballbearings, return_type="dataframe")
model_5a = LinearRegression(fit_intercept=False).fit(X_5a, y_5)
print(model_5a.score(X_5a, y_5))
