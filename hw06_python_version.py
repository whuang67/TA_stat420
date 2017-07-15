# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 19:02:15 2017

@author: whuang67
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from patsy import dmatrices
import matplotlib.pyplot as plt
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

# Question 1
## (a)
def fitted_vs_resid(model, X, y, pointcol = "blue", linecol = "green"):
    fitted = model.predict(X)
    resid = y - fitted
    plt.scatter(fitted, resid, color = pointcol)
    plt.axhline(y = 0, color = linecol)
    plt.show()


## (b)
def Normal_QQ(model, X, y):
    resid = (y - model.predict(X))
    resid = resid[resid.columns[0]]
    ss.probplot(resid, dist="norm", plot=plt)
    plt.show()
    
## (c)
test_data = pd.DataFrame({"x": np.random.uniform(0, 10, 20),
                          "y": np.array([0]*20)})
test_data.y = 5+2*test_data.x + np.random.normal(0, 1, 20)
y_1, X_1 = dmatrices("y ~ x", test_data, return_type = "dataframe")
model_1 = LinearRegression(fit_intercept=False).fit(X_1, y_1)
fitted_vs_resid(model_1, X_1, y_1)
Normal_QQ(model_1, X_1, y_1)



# Question 2
## (a)
os.chdir("C:/Users/whuang67/downloads")
swiss = pd.read_csv("swiss.csv")
y_2, X_2a = dmatrices(
        "Fertility ~ Agriculture + Examination + Education + Catholic + Infant_Mortality",
        swiss, return_type="dataframe")
model_2a = LinearRegression(fit_intercept=False).fit(X_2a, y_2)
print model_2a.coef_[0]

## (b)
fitted_vs_resid(model_2a, X_2a, y_2)

## (c)
Normal_QQ(model_2a, X_2a, y_2)

## (d)
H = np.dot(np.dot(X_2a,
                  np.linalg.inv(np.dot(np.transpose(X_2a),
                                       X_2a))),
                  np.transpose(X_2a))
Leverage = np.diag(H).tolist()

## (e)
resid_2 = (y_2-model_2a.predict(X_2a)).Fertility.tolist()
s2_p = sum(np.array(resid_2)**2)/(X_2a.shape[0]-X_2a.shape[1])*X_2a.shape[1]
Cooks_Distance = []
for e, h in zip(resid_2, Leverage):
    d = e**2/s2_p*(h/(1-h)**2)
    Cooks_Distance.append(d)
# Cooks_Distance
print np.where(np.array(Cooks_Distance) > 4.0/X_2a.shape[0])

## (f)
swiss_small = swiss.drop(swiss.index[[5, 36, 41, 45, 46]])
y_2f, X_2f = dmatrices(
        "Fertility ~ Agriculture + Examination + Education + Catholic + Infant_Mortality",
        swiss_small, return_type="dataframe")
model_2f = LinearRegression(fit_intercept=False).fit(X_2f, y_2f)
print model_2f.coef_

## (g)
influ = swiss.iloc[[5, 36, 41, 45, 46], :]
_, Pred = dmatrices(
        "Fertility ~ Agriculture + Examination + Education + Catholic + Infant_Mortality",
        influ, return_type="dataframe")
print model_2a.predict(Pred)
print model_2f.predict(Pred)



# Question 3
## (a)
tvdoctor = pd.read_csv("tvdoctor.csv")
y_3, X_3a = dmatrices("life ~ tv", tvdoctor, return_type = "dataframe")
model_3a = LinearRegression(fit_intercept=False).fit(X_3a, y_3)
abline_values = [model_3a.coef_[0][0] + model_3a.coef_[0][1]*x for x in sorted(tvdoctor.tv)]
plt.plot(sorted(tvdoctor.tv), abline_values)
plt.scatter(tvdoctor.tv, tvdoctor.life)
plt.show()
fitted_vs_resid(model_3a, X_3a, y_3)
Normal_QQ(model_3a, X_3a, y_3)

## (b)
_, X_3b3 = dmatrices("life ~ tv+I(tv**2)+I(tv**3)",
                       tvdoctor,
                       return_type="dataframe")
model_3b3 = LinearRegression(fit_intercept=False).fit(X_3b3, y_3)
fitted_vs_resid(model_3b3, X_3b3, y_3)

_, X_3b5 = dmatrices(
        "life ~ tv+I(tv**2)+I(tv**3)+I(tv**4)+I(tv**5)",
        tvdoctor, return_type= "dataframe")
model_3b5 = LinearRegression(fit_intercept=False).fit(X_3b5, y_3)
fitted_vs_resid(model_3b5, X_3b5, y_3)

_, X_3b7 = dmatrices(
        "life ~ tv+I(tv**2)+I(tv**3)+I(tv**4)+I(tv**5)+I(tv**6)+I(tv**7)",
        tvdoctor, return_type= "dataframe")
model_3b7 = LinearRegression(fit_intercept=False).fit(X_3b7, y_3)
fitted_vs_resid(model_3b7, X_3b7, y_3)
model_3b7.coef_

ANOVA(model_3b3, model_3b5, X_3b3, X_3b5, y_3)
ANOVA(model_3b3, model_3b7, X_3b3, X_3b7, y_3)
ANOVA(model_3b5, model_3b7, X_3b5, X_3b7, y_3)
Normal_QQ(model_3b5, X_3b5, y_3)



# Question 4
## (a)
mammals = pd.read_csv("mammals.csv")
print min(mammals.body)
print max(mammals.body)

## (b)
print min(mammals.brain)
print max(mammals.brain)

## (c)
plt.scatter(mammals.body, mammals.brain)
plt.show()

## (d)
y_4, X_4d = dmatrices("brain ~ body", mammals, return_type="dataframe")
model_4d = LinearRegression(fit_intercept=False).fit(X_4d, y_4)
print ANOVA_1(model_4d, X_4d, y_4)
fitted_vs_resid(model_4d, X_4d, y_4)
Normal_QQ(model_4d, X_4d, y_4)

## (e)
_, X_4e = dmatrices("brain ~ np.log(body)", mammals, return_type="dataframe")
model_4e = LinearRegression(fit_intercept=False).fit(X_4e, y_4)

## (f)
y_4f, X_4f = dmatrices("np.log(brain) ~ np.log(body)",
                       mammals, return_type="dataframe")
model_4f = LinearRegression(fit_intercept=False).fit(X_4f, y_4f)
abline_values = [model_4f.coef_[0][0] + model_4f.coef_[0][1]*x \
                 for x in sorted(np.log(mammals.body))]
plt.scatter(np.log(mammals.body), np.log(mammals.brain))
plt.plot(sorted(np.log(mammals.body)), abline_values)
plt.show()

## (g)
Normal_QQ(model_4f, X_4f, y_4f)

## (f)
print np.exp(model_4f.coef_[0][0] + model_4f.coef_[0][1]*np.log(13.4/2.2))



# Question 5
## (a)
epa2015 = pd.read_csv("epa2015.csv")
y_5, X_5a = dmatrices("CO2 ~ horse*type", epa2015, return_type="dataframe")
model_5a = LinearRegression(fit_intercept=False).fit(X_5a, y_5)
fitted_vs_resid(model_5a, X_5a, y_5)

## (b)
y_5b, X_5b = dmatrices("np.log(CO2) ~ horse*type",
                       epa2015, return_type="dataframe")
model_5b = LinearRegression(fit_intercept=False).fit(X_5b, y_5b)
fitted_vs_resid(model_5b, X_5b, y_5b)

## (c)
_, X_5c = dmatrices("np.log(CO2) ~ horse*type + I(horse**2)",
                       epa2015, return_type="dataframe")
model_5c = LinearRegression(fit_intercept=False).fit(X_5c, y_5b)
fitted_vs_resid(model_5c, X_5c, y_5b)

## (d)
Normal_QQ(model_5c, X_5c, y_5b)



# Question 6
## (a)
np.random.seed(6)   
n = 50
x_1 = np.random.uniform(0, 10, n)
x_2 = np.random.uniform(-5, 5, n)

p_vals_1 = []
p_vals_2 = []

for i in range(1000):
    y_1 = 2 + x_1 + 0*x_2 + np.random.normal(0, 1, n)
    y_2 = 2 + x_1 + 0*x_2 + np.random.normal(0, np.absolute(x_2), n)
    df = pd.DataFrame({"y_1": y_1,
                       "y_2": y_2,
                       "x_1": x_1,
                       "x_2": x_2})
    y_61, X_6 = dmatrices("y_1 ~ x_1+x_2", df, return_type="dataframe")
    y_62, _ = dmatrices("y_2 ~ x_1+x_2", df, return_type="dataframe")
    model_61 = LinearRegression(fit_intercept=False).fit(X_6, y_61)
    model_62 = LinearRegression(fit_intercept=False).fit(X_6, y_62)
    resid_1 = y_61 - model_61.predict(X_6)
    resid_2 = y_62 - model_62.predict(X_6)
    p_vals_1.append(ss.shapiro(resid_1)[1])
    p_vals_2.append(ss.shapiro(resid_2)[1])

## (b)
print np.mean(np.array(p_vals_1) < 0.05)
print np.mean(np.array(p_vals_1) < 0.1)
print np.mean(np.array(p_vals_2) < 0.05)
print np.mean(np.array(p_vals_2) < 0.1)


# Question 7
## (a)
np.random.seed(6)
n = 40
x = np.random.uniform(0, 10, n)

rmse_slr = []
rmse_big = []
pval = []
for i in range(1000):
    y = 3 - 4*x + np.random.normal(0, 3, n)
    df = pd.DataFrame({"y": y,
                       "x": x,
                       "x_2": np.power(x, 2)})
    y_1, X_1 = dmatrices("y~x", df, return_type="dataframe")
    _, X_2 = dmatrices("y~x+x_2", df, return_type="dataframe")
    model_1 = LinearRegression(fit_intercept=False).fit(X_1, y_1)
    model_2 = LinearRegression(fit_intercept=False).fit(X_2, y_1)
    
    rmse_slr.append(((y_1-model_1.predict(X_1))**2).mean()[0])
    rmse_big.append(((y_1-model_2.predict(X_2))**2).mean()[0])
    pval.append(ANOVA(model_1, model_2, X_1, X_2, y_1)['p_val'])

## (b)
print np.mean(np.array(rmse_slr) < np.array(rmse_big))

## (c)
print np.mean(np.array(pval) < 0.05)