# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 20:35:39 2017

@author: whuang67
"""

import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt

# Question 1
## (a)
x0 = np.array([1]*25)
x1 = np.arange(1, 26)**2
x2 = np.linspace(0, 1, 25)
x3 = np.log(np.arange(1, 26))
np.random.seed(42)
y = 5 * x0 + 1 * x1 + 6 * x2 + 3 * x3 + np.random.normal(0, 1, 25)
print sum(y)
## (b)
X = np.column_stack((x0, x1, x2, x3))
print X.sum()
## (c)
beta_hat = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)),
                         np.transpose(X)), y)
print beta_hat
## (d)
(beta_hat[1:4]**2).sum()
## (e)
y_hat = np.dot(X, beta_hat)
print ((y-y_hat)**2).sum()


# Question 2
## (a)
Binom = ss.binom(40, 0.12)
print Binom.pmf(5)
## (b)
total_p_2b = 0
for k in range(0, 11):
    total_p_2b = total_p_2b + Binom.pmf(k)
print total_p_2b
## (c)
total_p_2c = 0
Binom2 = ss.binom(40, .88)
for k in range(37, 41):
    total_p_2c = total_p_2c + Binom2.pmf(k)
print total_p_2c
## (d)
total_p_2d = 0
for k in range(3, 10):
    total_p_2d += Binom.pmf(k)
print total_p_2d


# Question 3
## (a)
Normal = ss.norm(100, 15)
print Normal.cdf(90)
## (b)
print 1-Normal.cdf(105)
## (c)
print Normal.cdf(100)-Normal.cdf(95)
## (d)
print Normal.ppf(.1)
## (e)
print Normal.ppf(1-.05)


# Question 4
## (a)
def do_t_test(x, mu = 0):
    import numpy as np
    n = len(y)
    t = (np.mean(y)-mu)/np.std(y, ddof=1)*n**.5
    pval = ss.t.sf(abs(t), n-1)*2
    return [t, pval]
## (b)
def make_decision(pval, alpha =.05):
    import numpy as np
    decision = np.where(pval <.05, "Reject!", "Fail to Reject.")
    return decision
## (c)
np.random.seed(42)
y = np.random.normal(1.4, 1, 25)
p_val = do_t_test(y, 2)[1]
print p_val
print make_decision(p_val, .1)


# Question 5
## (a)
file = "C:/users/whuang67/downloads/intelligence.csv"
intelligence = pd.read_csv(file)
## (b)
pawnee = intelligence.iq[intelligence.town == "pawnee"].values
eagleton = intelligence.iq[intelligence.town == "eagleton"].values
plt.boxplot([pawnee, eagleton])
plt.title("IQ vs Town of Origin")
plt.ylabel("Intelligence Quotient (IQ)")
plt.xticks([1, 2], ['pawnee', 'eagleton'])
plt.show()
## (c)
print ss.stats.ttest_ind(eagleton, pawnee, equal_var=False)[1]/2


# Question 6
## (a)
file = "C:/users/whuang67/downloads/diabetes.csv"
diabetes = pd.read_csv(file, index_col = 0).reset_index(drop = True)
## (b)
print diabetes.shape
## (c)
print diabetes.columns.values
## (d)
diabetes.hdl.mean()
## (e)
diabetes.chol.std(ddof = 1)
## (f)
print diabetes.age.min()
print diabetes.age.max()
## (g)
female = diabetes.hdl[diabetes.gender == "female"].values
print female.mean()
## (h)
plt.scatter(diabetes.weight, diabetes.hdl)
plt.title("HDL vs Weight")
plt.show()
## (i)
plt.scatter(diabetes.weight, diabetes.chol, color = "orange")
plt.title("Total Cholesterol vs Weight")
plt.show()
## (j)
male = diabetes.hdl[diabetes.gender == "male"].values
print ss.stats.ttest_ind(male, female, equal_var=False, nan_policy = "omit")