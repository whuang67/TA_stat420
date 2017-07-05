# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 14:01:49 2017

@author: whuang67
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.stats as ss
import matplotlib.pyplot as plt

# Questions 1
## (a)
faithful = pd.read_csv("C:/users/whuang67/downloads/faithful.csv",
                       index_col=0)
faithful.sort_values(by='waiting', inplace=True)
faithful.reset_index(drop = True)
X = faithful.drop("eruptions", axis=1)
y = faithful.eruptions
regr1 = LinearRegression().fit(X, y)
df = faithful.shape[0]-2
Sxx = ((X-X.mean())**2).sum()[0]
se = (sum((y - regr1.predict(X))**2)/df)**.5
se_beta1 = se/Sxx**.5
se_beta0 = se*(1.0/faithful.shape[0]+(X.mean()[0])**2/Sxx)**.5
t_beta1 = (regr1.coef_[0]-0)/se_beta1
p_val_beta1 = ss.t.sf(abs(t_beta1), df)*2
t_beta0 = (regr1.intercept_-0)/se_beta0
p_val_beta0 = ss.t.sf(abs(t_beta0), df)*2
print "Beta 0, t-value: {:.2f}, p-value: {:.4f}.".format(t_beta0, p_val_beta0)
print "Beta 1, t-value: {:.2f}, p-value: {:.4f}.".format(t_beta1, p_val_beta1)

## (b)
c_val = ss.t.ppf(1-.01/2, df)
beta1_ci = [regr1.coef_[0]-c_val*se_beta1, regr1.coef_[0]+c_val*se_beta1]
print "99% Confidence Interval for Beta1: {}".format(beta1_ci)

## (c)
c_val = ss.t.ppf(1-.1/2, df)
beta0_ci = [regr1.intercept_-c_val*se_beta0, regr1.intercept_+c_val*se_beta0]
print "90% Confidence Interval for Beta1: {}".format(beta0_ci)

## (d)
def CI_y(x, alpha = .05):
    se_y = se*(1.0/faithful.shape[0]+(x-X.mean()[0])**2/Sxx)**.5
    c_val = ss.t.ppf(1-alpha/2, df)
    y_hat = regr1.predict(x)[0]
    return [y_hat-c_val*se_y, y_hat+c_val*se_y]
print "95% Confidence Interval for 75 is {}.".format(CI_y(75))
print "95% Confidence Interval for 80 is {}.".format(CI_y(80))

## (e)
def Pred_y(x, alpha = .05):
    se_y_e = se*(1+1.0/faithful.shape[0]+(x-X.mean()[0])**2/Sxx)**.5
    c_val = ss.t.ppf(1-alpha/2, df)
    y_hat = regr1.predict(x)[0]
    return [y_hat-c_val*se_y_e, y_hat+c_val*se_y_e]
print "95% Prediction Interval for 75 is {}.".format(Pred_y(75))
print "95% Prediction Interval for 100 is {}.".format(Pred_y(100))

## (f)
pred_lwr, pred_upr, conf_lwr, conf_upr = [], [], [], []
for x in X.waiting.values:
    pred_lwr.append(Pred_y(x)[0])
    pred_upr.append(Pred_y(x)[1])
    conf_lwr.append(CI_y(x)[0])
    conf_upr.append(CI_y(x)[1])
plt.scatter(X, y)
plt.plot(X, regr1.predict(X), color = "red")
plt.plot(X, pred_lwr, color = 'orange')
plt.plot(X, pred_upr, color = 'orange')
plt.plot(X, conf_lwr, color = "green")
plt.plot(X, conf_upr, color = "green")
plt.show()


# Question 2
## (a)
diabetes = pd.read_csv("C:/users/whuang67/downloads/diabetes.csv",
                       index_col = 0).reset_index(drop=True)
dat = diabetes[["weight", "chol"]].dropna()
regr2a = LinearRegression().fit(dat[["weight"]], dat.chol)
y_hat = regr2a.predict(dat[['weight']])
SSM = sum((y_hat-np.mean(dat.chol))**2)
SSE = sum((dat.chol-y_hat)**2)
MSM = SSM/1
MSE = SSE/(dat.shape[0]-2)
F_stat = MSM/MSE
p_val = ss.f.sf(F_stat, 1, dat.shape[0]-2)
print "F_stat: {}".format(F_stat)
print "p_value: {}".format(p_val)

## (b)
dat = diabetes[['hdl', 'weight']].dropna()
regr2b = LinearRegression().fit(dat[['weight']], dat.hdl)
y_hat = regr2b.predict(dat[['weight']])
SSM = sum((y_hat-np.mean(dat.hdl))**2)
SSE = sum((dat.hdl-y_hat)**2)
MSM = SSM/1
MSE = SSE/(dat.shape[0]-2)
F_stat = MSM/MSE
p_val = ss.f.sf(F_stat, 1, dat.shape[0]-2)
print "F_stat: {}".format(F_stat)
print "p_value: {}".format(p_val)

# Question 3
goalies2017 = pd.read_csv("C:/users/whuang67/downloads/goalies2017.csv")
dat = goalies2017[["MIN", "W"]].dropna()
regr3 = LinearRegression().fit(dat[["MIN"]], dat.W)
beta_1 = regr3.coef_[0]
df = dat.shape[0]-2
se = (sum((dat.W - regr3.predict(dat[["MIN"]]))**2)/df)**.5
Sxx = ((dat[["MIN"]]-dat[["MIN"]].sum()/748)**2).sum()[0]
se_beta_1 = se/Sxx**.5
t_test_value = (beta_1-0.008)/se_beta_1
p_val = ss.t.cdf(t_test_value, df)
print "beta_1 is {:.4f}.".format(beta_1)
print "se_beta_1 is {}.".format(se_beta_1)
print "t_test_value is {:.5f}.".format(t_test_value)
print "Degrees of Freedom is {}.".format(df)
print "p-value is {:.8f}.".format(p_val)


# Question 4
## (a)
n = 50
np.random.seed(671105713)
x = np.linspace(0, 20, n)
beta_1s = []
beta_0s = []
for i in range(0, 1500):
    epsilon = np.random.normal(0, 5, n)
    y = 4 + 0.5*x + epsilon
    beta_1 = LinearRegression().fit(pd.DataFrame(x), y).coef_[0]
    beta_0 = LinearRegression().fit(pd.DataFrame(x), y).intercept_
    beta_1s.append(beta_1)
    beta_0s.append(beta_0)
## (b)
## (c)
print "Std of beta_1: {}".format(5/(((x-x.mean())**2).sum())**.5)
## (d)
print "Mean of simulated values of beta_1: {}".format(np.mean(beta_1s))
## (e)
print "Std of simulated values of beta_1: {}".format(np.std(beta_1s, ddof = 1))
## (f)
## (g)
print "Std of beta_0: {}".format(5*(1.0/n+(np.mean(x))**2/sum((x-np.mean(x))**2))**.5)
## (h)
print "Mean of simulated values of beta_0: {}".format(np.mean(beta_0s))
## (i)
print "Std of simulated values of beta_0: {}".format(np.std(beta_0s, ddof = 1))

## (j)
plt.hist(beta_1s, 25, normed=True)
fit_normal = ss.norm.pdf(sorted(beta_1s), .5, 5/(((x-x.mean())**2).sum())**.5)
plt.plot(sorted(beta_1s), fit_normal)
plt.show()

## (k)
plt.hist(beta_0s, 25, normed=True)
fit_normal = ss.norm.pdf(sorted(beta_0s),
                         4,
                         5*(1.0/n+(np.mean(x))**2/sum((x-np.mean(x))**2))**.5)
plt.plot(sorted(beta_0s), fit_normal)
plt.show()

## (l)
n = 50
np.random.seed(671105713)
x = np.linspace(0, 20, n)
for i in range(0, 100):
    epsilon = np.random.normal(0, 5, n)
    y = 4 + 0.5*x + epsilon
    beta_1 = LinearRegression().fit(pd.DataFrame(x), y).coef_[0]
    beta_0 = LinearRegression().fit(pd.DataFrame(x), y).intercept_
    plt.scatter(x, x*beta_1+beta_0, s=1, color="black")
plt.plot(x, 4+.5*x, color="red")
plt.show()



# Question 5
## (a)
n = 20
np.random.seed(671105713)
x = np.linspace(-5, 5, n)
se_beta_0s = []
beta_0s = []
for i in range(0, 2000):
    epsilon = np.random.normal(0, 4, n)
    y = 1 + 3*x + epsilon
    regr = LinearRegression().fit(pd.DataFrame(x), y)
    beta_0 = regr.intercept_
    beta_0s.append(beta_0)
    se = (sum((y - regr.predict(pd.DataFrame(x)))**2)/18.0)**.5
    se_beta_0 = se*(1.0/n+(np.mean(x))**2/sum((x-np.mean(x))**2))**.5
    se_beta_0s.append(se_beta_0)

## (b)
crit = ss.t.ppf(1-.1/2, n-2)
beta_0s = np.array(beta_0s)
se_beta_0s = np.array(se_beta_0s)
lower_90 = beta_0s-crit*se_beta_0s
upper_90 = beta_0s+crit*se_beta_0s

## (c)
percent90 = np.mean((lower_90 <= 1) & (upper_90 >=1))
print "{:.4f} contains the true value of beta_0.".format(percent90)

## (d)
reject90 = 1-np.mean((lower_90 <= 1) & (upper_90 >=1))
print "{:.4f} would reject H0: beta_0 = 0.".format(reject90)

## (e)
crit99 = ss.t.ppf(1-.01/2, n-2)
lower_99 = beta_0s-crit99*se_beta_0s
upper_99 = beta_0s+crit99*se_beta_0s

## (f)
reject99 = 1-np.mean((lower_90 <= 1) & (upper_90 >=1))
print "{:.4f} would reject H0: beta_0 = 0.".format(reject99)
percent99 = np.mean((lower_99 <= 1) & (upper_99 >=1))
print "{:.4f} contains the true value of beta_0.".format(percent99)
## (g)
