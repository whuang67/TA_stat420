# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 23:46:30 2017

@author: whuang67
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from patsy import dmatrices
import scipy.stats as ss
import matplotlib.pyplot as plt


def approximately_equal(first, second, tolerance = 10**-10):
    comparison = [abs(f - s) <= tolerance for f, s in zip(first, second)]
    if all(comparison):
        output = True
    else:
        output = False
    return output

abc = np.array([1, 2, 3])
bcd = np.array([1, 2, 3])
approximately_equal(abc, bcd)


longley = pd.read_csv("C:/users/whuang67/downloads/longley.csv")

a, W = dmatrices("GNP_deflator~ GNP + Unemployed+Year",
                 longley, return_type = "dataframe")
del W['Intercept']


class Linear_Regression():
    def __init__(self, X, y, summary = False, ANOVA = False, outlier= False):
        self.X = X*1
        self.y = y*1
        beta_s = np.matmul(
                np.matmul(
                    np.linalg.inv(
                        np.matmul(
                            np.transpose(self.X), self.X)),
                            np.transpose(self.X)), self.y)
        self.coef_ = {
                key: val[0] for key, val in zip(self.X.columns.tolist(),
                                             beta_s)}
        self.fitted = np.matmul(self.X, beta_s)
        self.resid = y - self.fitted
        
        ## Summary starts here!!! ###############################
        se = ((self.resid**2).sum()/(self.X.shape[0] - self.X.shape[1]))**.5
        if summary == True:
            
            C_diag = np.diag(np.linalg.inv(np.dot(np.transpose(self.X),self.X)))
            se_beta = se[0]*C_diag**.5
            self.t_stat = [(c - 0.0)/b for c, b in zip(self.coef_.values(), se_beta)]
            self.p_val = [2*ss.t.sf(abs(t), self.X.shape[0]-self.X.shape[1]) for t in self.t_stat]
            print("\nSummary:\n")
            for key, c, sd, t, p in zip(self.X.columns.tolist(),
                                        self.coef_.values(),
                                        se_beta,
                                        self.t_stat,
                                        self.p_val):
                print("{}:\nCoefficient: {:.5f}\nStd. Err: {:.5f}\nt_values: {:.3f}\nPr(>|t|): {:.4f}\n".format(key, c, sd, t, p))
        
        ## ANOVA starts here!!! (Significance of Regression) ######
        if ANOVA == True:
            SSR = ((self.fitted - self.y.mean().values)**2).sum()
            MSR = SSR/ (self.X.shape[1]-1)
            MSE = (se**2).values[0]
            self.F_stat = MSR/MSE
            self.F_p = ss.f.sf(self.F_stat, self.X.shape[1]-1, self.X.shape[0]-self.X.shape[1])

            print("\nAnalysis of Variance Table:\n")
            print("SSR: {:.2f} | DF_R: {:d} | MSR: {:.2f} | F: {:.2f}\nSSE: {:.2f} | DF_E: {:d} | MSE: {:.2f}".format(SSR,
                 self.X.shape[1]-1,MSR, self.F_stat, MSE*(self.X.shape[0]-self.X.shape[1]), self.X.shape[0]-self.X.shape[1],
                 MSE))
            print("F_statistic: {:.2f} on {:d} and {:d} DF\np_value: {:.4f}".format(self.F_stat, self.X.shape[1]-1, self.X.shape[0]-self.X.shape[1], self.F_p))
        
        ## Outlier test starts here!!! ############################
        if outlier == True:
            def get_Leverage(X):
                H = np.matmul(np.matmul(X, np.linalg.inv(np.matmul(np.transpose(X), X))),
                              np.transpose(X))
                h_s = np.diag(H)
                return(h_s)
            
            self.leverage = get_Leverage(self.X).tolist()
            s2p = (se**2*self.X.shape[1]).values[0]
            self.cooksdist = [e[0]**2/s2p*h/(1.0-h)**2 \
                              for e, h in zip(self.resid.values, self.leverage)]
            print("\nHigh Leverage Points: ")
            print([i for i, point in enumerate(self.leverage) \
                   if point > 2*sum(self.leverage) / self.X.shape[0]])
            print("\nInfluential Points (Cook's distance):")
            print([i for i, point in enumerate(self.cooksdist) \
                   if point > 4 / self.X.shape[0]])
        
## Assumption test starts here!!! ############################
def fitted_vs_resid(linear_model, pointcol = "blue", linecol = "red"):
    fitted = linear_model.fitted
    resid = linear_model.resid
    plt.scatter(fitted, resid, color = pointcol)
    plt.axhline(y = 0, color = linecol)
    plt.title("Fitted vs Residuals Plot")

def Normal_QQ(linear_model):
    ss.probplot([e[0] for e in linear_model.resid.values], dist='norm', plot=plt)

## Analysis of Variance #####
class ANOVA():
    def __init__(self, model_reduced, model_full, trace = True):
        SSE_reduced = (model_reduced.resid**2).sum().values[0]
        SSE_full = (model_full.resid**2).sum().values[0]
        df_reduced = model_reduced.X.shape[0] - model_reduced.X.shape[1]
        df_full = model_full.X.shape[0] - model_full.X.shape[1]
        denominator = SSE_full/df_full
        numerator = (SSE_reduced - SSE_full)/(df_reduced - df_full)
        self.F_stat = (numerator / denominator)
        self.p_val = ss.f.sf(self.F_stat, df_reduced-df_full, df_full)
        if trace == True:
            print("\nAnalysis of Variance Table:\n")
            print("Model1 | residDF: {:d} | RSS: {:.4f}".format(df_reduced, SSE_reduced))
            print("Model2 | residDF: {:d} | RSS: {:.4f} | DF: {} | SumOfSq: {:.4f} | F: {:.2f}".format(df_full,
                  SSE_full, df_reduced-df_full, SSE_reduced-SSE_full, self.F_stat))
            print("F_statistic: {:.2f} on {:d} and {:d} DF\np_value: {:.4f}".format(self.F_stat,
                 df_reduced-df_full, df_full, self.p_val))


