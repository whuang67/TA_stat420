# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 23:46:36 2017

@author: whuang67
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Question 1
## (a)
file = "C:/users/whuang67/downloads/faithful.csv"
faithful = pd.read_csv(file, index_col = 0).reset_index(drop=True)
regr = LinearRegression()
X = faithful.drop("eruptions", axis=1)
y = faithful.eruptions
regr.fit(X, y)

##( b)
print regr.coef_
print regr.intercept_

## (c)
print regr.predict(80)

## (d)
print regr.predict(120)

## (e)
residuals = y - regr.predict(X)
print sum(residuals**2)

## (f)
plt.scatter(faithful.waiting, faithful.eruptions)
plt.plot(faithful.waiting*regr.coef_+regr.intercept_,
         color = "orange")
plt.show()

## (g)
print regr.score(X, y)


# Question 2
## (a)
def get_sd_est(model_resid, mle = False):
    if(mle == False):
        output = sum(model_resid**2)/(len(model_resid)-2)
    else:
        output = sum(model_resid**2)/len(model_resid)
    return output**.5

## (b)
print get_sd_est(residuals)

## (c)
print get_sd_est(residuals, mle = True)

## (d)


# Question 3
## (a)
def get_beta_no_int(x, y):
    beta = sum(x*y)/sum(x**2)
    return beta

## (b)
print get_beta_no_int(faithful.waiting, faithful.eruptions)

## (c)
regr3 = LinearRegression(fit_intercept = False).fit(X, y)
print regr3.coef_


# Question 4
## (a)
np.random.seed(1)
def sim_slr(n, beta_0, beta_1, sigma, xmin = 0, xmax = 10):
    import numpy as np
    import pandas as pd
    x = np.random.uniform(xmin, xmax, n)
    epsilon = np.random.normal(0, 2 , 50)
    y = beta_0 + beta_1 * x + epsilon
    output = pd.DataFrame({"predictor": x, "response": y})
    return output
data4 = sim_slr(50, beta_0 = 3, beta_1 = -7, sigma = 2)

## (b)
regr4 = LinearRegression().fit(data4.drop("response", axis=1),
                               data4.response)
print regr4.coef_
print regr4.intercept_

## (c)
plt.scatter(data4.predictor, data4.response)
plt.plot(data4.predictor,
         data4.predictor*regr4.coef_+regr4.intercept_,
         color = "orange")
plt.show()

## (d)
beta_hat_1 = []
for i in range(0, 2000):
    dat = sim_slr(50, beta_0 = 3, beta_1 = -7, sigma = 2)
    model = LinearRegression().fit(dat.drop("response", axis = 1), 
                                   dat.response)
    beta_hat_1.append(model.coef_[0])

## (e)
print np.array(beta_hat_1).mean()
print np.array(beta_hat_1).std(ddof = 1)

## (f)
plt.hist(beta_hat_1, 25)
plt.show()



# Question 5
## (a)
beta_hat_1 = []
for i in range(0, 1500):
    dat = sim_slr(50, beta_0 = 10, beta_1 = 0, sigma = 1)
    model = LinearRegression().fit(dat.drop("response", axis =1),
                                   dat.response)
    beta_hat_1.append(model.coef_[0])

## (b)
plt.hist(beta_hat_1, 25)
plt.show()

## (c)
file = "C:/users/whuang67/downloads/skeptic.csv"
skeptic = pd.read_csv(file)
regr5 = LinearRegression().fit(skeptic.drop('response', axis=1),
                               skeptic.response)

## (d)
plt.hist(beta_hat_1, 25)
plt.axvline(x = regr5.coef_, color = "red")
plt.show()

## (e)
print (beta_hat_1 > regr5.coef_).mean()


# Questoin 6
## (a)
file = "C:/users/whuang67/downloads/goalies2017.csv"
goalies2017 = pd.read_csv(file)
dat = goalies2017[["MIN", "W"]].dropna()
regr61 = LinearRegression().fit(dat.MIN.to_frame(),
                                dat.W)
def get_RMSE(y_true, y_pred):
    RMSE = np.mean((y_true - y_pred)**2)**.5
    return RMSE
print get_RMSE(dat.W, regr61.predict(dat.MIN.to_frame()))
plt.scatter(dat.MIN, dat.W)
plt.plot(dat.MIN, dat.MIN*regr61.coef_+regr61.intercept_, color = "red")
plt.show()

## (b)
dat = goalies2017[["GA", "W"]].dropna()
regr62 = LinearRegression().fit(dat.GA.to_frame(),
                                dat.W)
print get_RMSE(dat.W, regr62.predict(dat.GA.to_frame()))
plt.scatter(dat.GA, dat.W)
plt.plot(dat.GA, dat.GA*regr62.coef_+regr62.intercept_, color = "red")
plt.show()

## (c)
dat = goalies2017[["SO", "W"]].dropna()
regr63 = LinearRegression().fit(dat.SO.to_frame(),
                                dat.W)
print get_RMSE(dat.W, regr63.predict(dat.SO.to_frame()))
plt.scatter(dat.SO, dat.W)
plt.plot(dat.SO, dat.SO*regr63.coef_+regr63.intercept_, color = "red")
plt.show()