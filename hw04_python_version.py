# -*- coding: utf-8 -*-
"""
Created on Sat Jul 01 21:36:06 2017

@author: Wenke Huang
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import scipy.stats as ss

# Question 1
## (a)
path = "C:/users/whuang67/downloads/nutrition.csv"
nutrition = pd.read_csv(path)
nutrition.drop(nutrition[['ID', 'Desc', 'Portion']], axis=1, inplace=True)
X_1a = nutrition.drop(nutrition[['Calories']], axis=1)
y_1 = nutrition.Calories
regr_1a = LinearRegression().fit(X_1a, y_1)
SSR = ((regr_1a.predict(X_1a)-y_1.mean())**2).sum()
df_R = X_1a.shape[1]
MSR = SSR/df_R
SSE = ((y_1-regr_1a.predict(X_1a))**2).sum()
df_E = X_1a.shape[0]-X_1a.shape[1]-1
MSE = SSE/df_E
F_stat = MSR/MSE
p_val = ss.f.sf(F_stat, df_R, df_E)
print "F-stat is {}.".format(F_stat)
print "P-value is {}.".format(p_val)

## (b)
## (c)
X_1c = nutrition[['Carbs', 'Sodium', 'Fat', 'Protein']]
regr_1c = LinearRegression().fit(X_1c, y_1)
SSR = ((regr_1c.predict(X_1c)-y_1.mean())**2).sum()
df_R = X_1c.shape[1]
MSR = SSR/df_R
SSE = ((y_1-regr_1c.predict(X_1c))**2).sum()
df_E = X_1c.shape[0]-X_1c.shape[1]-1
MSE = SSE/df_E
F_stat = MSR/MSE
p_val = ss.f.sf(F_stat, df_R, df_E)
print "F-stat is {}.".format(F_stat)
print "P-value is {}.".format(p_val)

## (d)

# Question 2
## (a)
X_2a = nutrition[['Carbs', 'Fat', 'Protein']]
regr_2a = LinearRegression().fit(X_2a, y_1)
SSR = ((regr_2a.predict(X_2a)-y_1.mean())**2).sum()
df_R = X_2a.shape[1]
MSR = SSR/df_R
SSE = ((y_1-regr_2a.predict(X_2a))**2).sum()
df_E = X_2a.shape[0]-X_2a.shape[1]-1
MSE = SSE/df_E
F_stat = MSR/MSE
p_val = ss.f.sf(F_stat, df_R, df_E)
print "F-stat is {}.".format(F_stat)
print "P-value is {}.".format(p_val)

## (b)
print(regr_2a.coef_)
print(regr_2a.intercept_)
## (c)
regr_2a.predict(pd.DataFrame({'Carbs': [47],
                              'Fat': [28],
                              'Protein': [25]}))
## (d)
regr_2a.predict(pd.DataFrame({'Carbs': [47],
                              'Fat': [28],
                              'Protein': [25]}))