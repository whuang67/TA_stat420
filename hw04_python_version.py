# -*- coding: utf-8 -*-
"""
Created on Sat Jul 05 00:01:06 2017
@author: Wenke Huang
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.stats as ss
import matplotlib.pyplot as plt
# from IPython.display import display

def ANOVA(model, X, y):
    SSR = ((model.predict(X)-y.mean())**2).sum()
    df_R = X.shape[1]
    MSR = SSR/df_R
    SSE = ((y-model.predict(X))**2).sum()
    df_E = X.shape[0]-X.shape[1]-1
    MSE = SSE/df_E
    F_stat = MSR/MSE
    p_of_F = ss.f.sf(F_stat, df_R, df_E)
    return({'SSR': SSR,
            'df_R': df_R,
            'MSR': MSR,
            'SSE': SSE,
            'df_E': df_E,
            'MSE': MSE,
            'F_stat': F_stat,
            'p_value': p_of_F})

def get_RMSE(model, X, y):
    resid = y-model.predict(X)
    RMSE = ((resid**2).mean())**.5
    return(RMSE)

def Summary(model, X, y):
    se = (((y-model.predict(X))**2).sum()/(X.shape[0]-X.shape[1]-1))**.5
    X_1 = X.copy()
    X_1['Intercept'] = pd.Series(np.array([1]*(X_1.shape[0]+1)))
    C_diag = np.diag(np.linalg.inv(np.dot(np.transpose(X_1), X_1)))
    coefficients = np.append(model.coef_, model.intercept_)
    se_beta = se*C_diag**.5
    t_stat = (coefficients-0.0)/se_beta
    p_of_t = 2*ss.t.sf(abs(t_stat), X.shape[0]-X.shape[1]-1)
    result = pd.DataFrame({"Coefficients": coefficients,
                           "Std_Error":se_beta,
                           "t_value": t_stat,
                           "p_value": p_of_t},
                          index = np.append(X.columns, "(Intercept)"))
    return(result)

def confint(model, X, y, level=0.95):
    se = (((y-model.predict(X))**2).sum()/(X.shape[0]-X.shape[1]-1))**.5
    X_1 = X.copy()
    X_1['Intercept'] = pd.Series(np.array([1]*(X_1.shape[0]+1)))
    C_diag = np.diag(np.linalg.inv(np.dot(np.transpose(X_1), X_1)))
    coefficients = np.append(model.coef_, model.intercept_)
    se_beta = se*C_diag**.5
    crit_val = ss.t.ppf(.5+level/2, X.shape[0]-X.shape[1]-1)
    lower = coefficients-crit_val*se_beta
    upper = coefficients+crit_val*se_beta
    result = pd.DataFrame({"Coefficients": coefficients,
                           "Lower_bound": lower,
                           "Upper_bound": upper},
                          index = np.append(X.columns, "(Intercept)"))
    return(result)

def CI_y(model, X, y, x, level = 0.95, interval = "confidence"):
    X_1 = X.copy()
    X_1['Intercept'] = pd.Series(np.array([1]*(X_1.shape[0]+1)))
    x_1 = x.copy()
    x_1['Intercpet'] = pd.Series(np.array([1]))
    se = (((y-model.predict(X))**2).sum()/(X.shape[0]-X.shape[1]-1))**.5
    if interval == "confidence":
        se_y = np.dot(np.dot(x_1,
                             np.linalg.inv(np.dot(np.transpose(X_1),
                                                  X_1))),
                      np.transpose(x_1))**.5*se
    elif interval == "prediction":
        se_y = (np.dot(np.dot(x_1,
                              np.linalg.inv(np.dot(np.transpose(X_1),
                                                   X_1))),
                       np.transpose(x_1))+1)**.5*se
    c_val = ss.t.ppf(.5+level/2, X.shape[0]-X.shape[1])
    y_hat = model.predict(x)
    lower = y_hat-c_val*se_y
    upper = y_hat+c_val*se_y
    result = pd.DataFrame({"fit": y_hat,
                           "lower": lower[0],
                           "upper": upper[0]})
    return result




# Question 1
## (a)
path = "C:/users/whuang67/downloads/nutrition.csv"
nutrition = pd.read_csv(path)
nutrition.drop(nutrition[['ID', 'Desc', 'Portion']], axis=1, inplace=True)
X_1a = nutrition.drop(nutrition[['Calories']], axis=1)
y_1 = nutrition.Calories
regr_1a = LinearRegression().fit(X_1a, y_1)
result_1a = ANOVA(regr_1a, X_1a, y_1)
print "F-stat is {}.".format(result_1a['F_stat'])
print "P-value is {:.4f}.".format(result_1a['p_value'])
    
## (b)
print(Summary(regr_1a, X_1a, y_1))

## (c)
X_1c = nutrition[['Carbs', 'Sodium', 'Fat', 'Protein']]
regr_1c = LinearRegression().fit(X_1c, y_1)
result_1c = ANOVA(regr_1c, X_1c, y_1)
print "F-stat is {}.".format(result_1c['F_stat'])
print "P-value is {:.4f}.".format(result_1c['p_value'])

## (d)
print(Summary(regr_1c, X_1c, y_1))
## (e)




# Question 2
## (a)
X_2a = nutrition[['Carbs', 'Fat', 'Protein']]
regr_2a = LinearRegression().fit(X_2a, y_1)
result_2a = ANOVA(regr_2a, X_2a, y_1)
print "F-stat is {}.".format(result_2a["F_stat"])
print "P-value is {}.".format(result_2a["p_value"])

## (b)
print Summary(regr_2a, X_2a, y_1)[['Coefficients']]
## (c)
print regr_2a.predict(pd.DataFrame({'Carbs': [47],
                                    'Fat': [28],
                                    'Protein': [25]}))
    
## (d)
s_y = y_1.std()
s_e = (((y_1-regr_2a.predict(X_2a))**2).sum()/(X_2a.shape[0]-X_2a.shape[1]-1))**.5
print "s_y is {}.".format(s_y)
print "s_e is {}.".format(s_e)

## (e)
print "R^2 is {}.".format(regr_2a.score(X_2a, y_1))

## (f)
print confint(regr_2a, X_2a, y_1, level = .9)

## (g)
print confint(regr_2a, X_2a, y_1)

## (h)
print CI_y(regr_2a, X_2a, y_1, pd.DataFrame({'Carbs': [30],
                                             'Fat': [11],
                                             'Protein': [2]}), level = 0.99)

## (i)
print CI_y(regr_2a, X_2a, y_1, pd.DataFrame({'Carbs': [11],
                                             'Fat': [1.5],
                                             'Protein': [1]}), level = 0.9,
           interval = "prediction")




# Question 3
## (a)
path = "C:/users/whuang67/downloads/goalies_cleaned2015.csv"
goalies = pd.read_csv(path)
X_3a = goalies.drop(goalies[['W']], axis=1)
y_3 = goalies.W
regr3_full = LinearRegression().fit(X_3a, y_3)
result_3a = ANOVA(regr3_full, X_3a, y_3)
print "F-stat is {}.".format(result_3a['F_stat'])
print "P-value is {:.4f}.".format(result_3a['p_value'])

## (b)
RMSE_3b = get_RMSE(regr3_full, X_3a, y_3)
print "RMSE is of full model is {}.".format(RMSE_3b)

## (c)
X_3c = goalies[['GA', 'GAA', 'SV', 'SV_PCT']]
regr3_small = LinearRegression().fit(X_3c, y_3)
RMSE_3c = get_RMSE(regr3_small, X_3c, y_3)
print "RMSE of small model is {}.".format(RMSE_3c)

## (d)
X_3d = goalies[['GAA', 'SV_PCT']]
regr3_small2 = LinearRegression().fit(X_3d, y_3)
RMSE_3d = get_RMSE(regr3_small2, X_3d, y_3)
print "RMSE of small model is {}.".format(RMSE_3d)

## (e)
## (f)
SSE_reduced = ((y_3-regr3_small2.predict(X_3d))**2).sum()
SSE_full = ((y_3-regr3_small.predict(X_3c))**2).sum()
denominator = SSE_full/(X_3c.shape[0]-X_3c.shape[1]-1)
numerator = (SSE_reduced-SSE_full)/(X_3c.shape[1]-X_3d.shape[1])
F_stat = numerator/denominator
p_value = ss.f.sf(F_stat, X_3c.shape[1]-X_3d.shape[1], X_3c.shape[0]-X_3c.shape[1]-1)
print "F statistic is {}.".format(F_stat)
print "p value is {:.4f}.".format(p_value)




# Question 4
## (a)
np.random.seed(42)
n = 25
x0 = np.array([1]*n)
x1 = np.random.uniform(0, 10, n)
x2 = np.random.uniform(0, 10, n)
x3 = np.random.uniform(0, 10, n)
x4 = np.random.uniform(0, 10, n)
X = np.column_stack((x0, x1, x2, x3, x4))
C = np.linalg.inv(np.dot(np.transpose(X), X))
y = np.array([0]*n)
ex_4_data = pd.DataFrame({"y": y,
                          "x1": x1,
                          "x2": x2,
                          "x3": x3,
                          "x4": x4})
print np.diag(C)
print ex_4_data[9:10]

## (b)
beta_hat_1 = []
beta_2_pval = []
beta_3_pval = []

## (c)
np.random.seed(42)
for i in range(0, 1500):
    X = ex_4_data[['x1', 'x2', 'x3', 'x4']]
    y = 2+3*X.x1+4*X.x2+X.x4+np.random.normal(0, 4, n)
    regr = LinearRegression().fit(X, y)
    beta_hat_1.append(regr.coef_[0])
    p_values = Summary(regr, X, y).p_value.values
    beta_2_pval.append(p_values[1:2][0])
    beta_3_pval.append(p_values[2:3][0])

## (d)
## (e)
print np.mean(np.array(beta_hat_1))
print np.std(np.array(beta_hat_1), ddof=1)
plt.hist(beta_hat_1, 25, normed=True)
fit_normal = ss.norm.pdf(sorted(beta_hat_1),
                         3,
                         (16*np.diag(C)[1])**.5)
plt.plot(sorted(beta_hat_1), fit_normal)
plt.show()

## (f)
print np.mean(np.array(beta_3_pval) < 0.05)
## (g)
print np.mean(np.array(beta_2_pval) < 0.05)
