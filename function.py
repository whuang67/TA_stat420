# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 23:46:30 2017

@author: Wenke Huang (whuang67)
"""

import pandas as pd
import numpy as np
import os
import math
# from sklearn.linear_model import LinearRegression
from patsy import dmatrices
import scipy.stats as ss
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt



def Transformer(predictors, method = "z_score"):
    newdata = predictors.copy()
    ## Standard Scaler ## mean == 0, std == 1
    if method == "z_score":
        for column in newdata.columns:
            mean = newdata[column].mean()
            std = newdata[column].std()
            if std != 0:
                loc = mean
                scale = std
                newdata[column] = (newdata[column] - loc)/scale
    ## Min_Max Scaler ## range(0, 1)
    elif method == "min_max":
        for column in newdata.columns:
            min_ = newdata[column].min()
            max_ = newdata[column].max()
            if max_ != min_:
                newdata[column] = (newdata[column] - min_)/(max_ - min_)
            else:
                newdata[column] = newdata[column]/max_
    ## Abs_Max Scaler ## range(-1, 1)
    elif method == "max_abs":
        for column in newdata.columns:
            max_ = newdata[column].abs().max()
            newdata[column] = newdata[column]/max_
    ## Others ## Original dataset is returned
    return newdata


class Linear_Regression():
    def __init__(self, formula, data, summary = False):
        self.formula = formula
        self.data = data*1
        self.y, self.X = dmatrices(formula, self.data, return_type = "dataframe")
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
        self.resid = self.y - self.fitted

class summary():
    def __init__(self, model):
        ## Summary starts here!!! ###############################
        se = ((model.resid**2).sum()/(model.X.shape[0] - model.X.shape[1]))**.5
        C_diag = np.diag(np.linalg.inv(np.dot(np.transpose(model.X), model.X)))
        se_beta = se[0]*C_diag**.5
        self.t_stat = [(c - 0.0)/b for c, b in zip(model.coef_.values(), se_beta)]
        self.p_val = [2*ss.t.sf(abs(t), model.X.shape[0]-model.X.shape[1]) \
                      for t in self.t_stat]
        print("Summary:\n")
        for key, c, sd, t, p in zip(model.X.columns.tolist(),
                                    model.coef_.values(),
                                    se_beta,
                                    self.t_stat,
                                    self.p_val):
            print("{}:\nCoefficient: {:.5f}\nStd. Err: {:.5f}\nt_values: {:.3f}\nPr(>|t|): {:.4f}\n".format(key,
                  c, sd, t, p))

    ## ANOVA starts here!!! (Significance of Regression) ######
        SSR = ((model.fitted - model.y.mean().values)**2).sum()
        MSR = SSR/ (model.X.shape[1]-1)
        MSE = (se**2).values[0]
        SSE = MSE*(model.X.shape[0]-model.X.shape[1])
        self.F_stat = MSR/MSE
        self.F_p = ss.f.sf(self.F_stat, model.X.shape[1]-1, model.X.shape[0]-model.X.shape[1])
        self.R_2 = SSR / (SSR+SSE)
        Adjusted_R_2 = self.R_2 - (1-self.R_2)*(model.X.shape[1]-1)/(model.X.shape[0]-model.X.shape[1])
        print("Analysis of Variance Table:\n")
        print("SSR: {:.2f} | DF_R: {:d} | MSR: {:.2f} | F: {:.2f}\nSSE: {:.2f} | DF_E: {:d} | MSE: {:.2f}".format(SSR,
             model.X.shape[1]-1,MSR, self.F_stat, SSE, model.X.shape[0]-model.X.shape[1], MSE))
        print("F_statistic: {:.2f} on {:d} and {:d} DF\np_value: {:.4f}".format(self.F_stat,
              model.X.shape[1]-1, model.X.shape[0]-model.X.shape[1], self.F_p))
        print("\nR-squared: {:.4f}, Adjusted R-Sq: {:.4f}\nResidual standard error: {:.4f} on {:d} DF".format(self.R_2,
              Adjusted_R_2, se.values[0], model.X.shape[0]-model.X.shape[1]))

    ## Outlier test starts here!!! ############################

def get_Leverage(X):
    H = np.matmul(np.matmul(X, np.linalg.inv(np.matmul(np.transpose(X), X))),
                  np.transpose(X))
    h_s = np.diag(H)
    return(h_s)
class outlier_detection():
    def __init__(self, model):
        se = ((model.resid**2).sum()/(model.X.shape[0] - model.X.shape[1]))**.5
        self.leverage = get_Leverage(model.X).tolist()
        s2p = (se**2*model.X.shape[1]).values[0]
        self.cooksdist = [e[0]**2/s2p*h/(1.0-h)**2 \
                          for e, h in zip(model.resid.values, self.leverage)]
        print("\nHigh Leverage Points: ")
        print([i for i, point in enumerate(self.leverage) \
               if point > 2*sum(self.leverage) / model.X.shape[0]])
        print("\nInfluential Points (Cook's distance):")
        print([i for i, point in enumerate(self.cooksdist) \
               if point > 4 / model.X.shape[0]])
   
     
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
class get_ANOVA():
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

## Variance Inflation Factor
class get_VIF():
    def __init__(self, model, trace = True):
        ary = np.array(model.X)
        var_idx = [i for i, var in enumerate(model.X.columns) if var != "Intercept"]
        self.vif = [variance_inflation_factor(ary, j) for j in var_idx]
        print("Variance Inflation Factors:\n")
        for var, v in zip(var_idx, self.vif):
            print("{}: {}".format(model.X.columns[var], v))

def approximately_equal(first, second, tolerance = 10**-10):
    comparison = [abs(f - s) <= tolerance for f, s in zip(first, second)]
    if all(comparison):
        output = True
    else:
        output = False
    return output

def get_criteria(model, use = "AIC"):
    RSS = (model.resid**2).sum()[0]
    n = model.X.shape[0]
    p = model.X.shape[1]
    if use == "AIC":
        criteria = n*math.log(RSS/n) + 2*p
    elif use == "BIC":
        criteria = n*math.log(RSS/n) + math.log(n)*p
    return criteria

def feature_selection(model, method = "backward"):
    full_track = model.X.columns.tolist()
    full_track.remove("Intercept")
    left_vars = full_track*1
    init_formula = model.y.columns[0]+"~1"
    
    def backward_elimination(model):
        columns = model.X.columns.tolist()
        columns.remove("Intercept")
        criteria_dict = {'full': get_criteria(model)}
        print("Beginning Model: {}".format(criteria_dict['full']))
        for var in columns:
            formula = model.y.columns[0]+"~1"
            for var2 in columns:
                if var2 != var:
                   formula = formula + "+" + var2

            model_ = Linear_Regression(formula, model.data)
            criteria_dict[var] = get_criteria(model_)
            print("- {}: {}".format(var, criteria_dict[var]))
        mini, mini_key = criteria_dict["full"], 'full'
        for var in columns:
            if criteria_dict[var] < mini:
                mini = criteria_dict[var]
                mini_key = var
        if mini_key != 'full':
            formula_ = model.y.columns[0]+"~1"
            kept_vars = [v for v in columns if v != mini_key]
            left_vars.remove(mini_key)
            print("We choose to remove: {}\n".format(mini_key))
            for var in kept_vars:
                formula_ = formula_ + "+"+var
            model_2 = Linear_Regression(formula_, model.data)
            return backward_elimination(model_2)
        else:
            print("Finished!!! Variables {} were selected.".format(left_vars))
    
    def stepwise_regression(model):
        columns = model.X.columns.tolist()
        columns.remove("Intercept")
        criteria_dict = {'full': get_criteria(model)}
        print("Beginning Model: {}".format(criteria_dict['full']))
        diff_set = [v for v in full_track if v not in columns]
        for var in columns:
            formula = model.y.columns[0]+"~1"
            for var2 in columns:
                if var2 != var:
                   formula = formula + "+" + var2
            model_ = Linear_Regression(formula, model.data)
            criteria_dict[var] = get_criteria(model_)
            print("- {}: {}".format(var, criteria_dict[var]))
        for var in diff_set:
            formula = model.formula+ "+"+var
            model_ = Linear_Regression(formula, model.data)
            criteria_dict[var] = get_criteria(model_)
            print("+ {}: {}".format(var, criteria_dict[var]))
        mini, mini_key = criteria_dict["full"], 'full'
        for key, val in criteria_dict.items():
            if val < mini:
                mini = val
                mini_key = key
        
        if mini_key in columns:
            formula_ = model.y.columns[0]+"~1"
            kept_vars = [v for v in columns if v != mini_key]
            left_vars.remove(mini_key)
            print("We choose to remove: {}\n".format(mini_key))
            for var in kept_vars:
                formula_ = formula_ + "+"+var
            model_2 = Linear_Regression(formula_, model.data)
            return stepwise_regression(model_2)
        elif mini_key in diff_set:
            formula_ = model.formula + "+" + mini_key
            print("We choose to add: {}\n".format(mini_key))
            left_vars.append(mini_key)
            model_2 = Linear_Regression(formula_, model.data)
            return stepwise_regression(model_2)
        else:
            print("Finished!!! Variables {} were selected.".format(left_vars))
            
    def forward_selection(model, init_):
        model_for = Linear_Regression(init_, model.data)
        criteria_dict = {'full': get_criteria(model_for)}
        columns = model_for.X.columns.tolist()
        columns.remove("Intercept")
        print("Beginning Model: {}".format(criteria_dict['full']))
        diff_set = [v for v in full_track if v not in columns]
        for var in diff_set:
            formula = model_for.formula+ "+"+var
            model_ = Linear_Regression(formula, model.data)
            criteria_dict[var] = get_criteria(model_)
            print("+ {}: {}".format(var, criteria_dict[var]))
        mini, mini_key = criteria_dict["full"], 'full'
        for key, val in criteria_dict.items():
            if val < mini:
                mini = val
                mini_key = key
        
        if mini_key in diff_set:
            init_formula = model_for.formula + "+" + mini_key
            print("We choose to add: {}\n".format(mini_key))
            left_vars.append(mini_key)
            model_2 = Linear_Regression(init_formula, model.data)
            return forward_selection(model_2, init_ = init_formula)
        else:
            print("Finished!!! Variables {} were selected.".format(left_vars))
        
        
            ###########################
            ###########################
    if method == "backward":
        return backward_elimination(model)
    elif method == "stepwise":
        return stepwise_regression(model)
    elif method == "forward":
        return forward_selection(model, init_ = init_formula)
        




"""
Example!!!
"""

os.chdir("C:/Users/whuang67/downloads")
longley = pd.read_csv("longley.csv")
response, predictors = dmatrices("GNP_deflator~ GNP + Unemployed + Year",
                                 longley, return_type = "dataframe")

## Model Fitting
model_1 = Linear_Regression("GNP_deflator ~ GNP+Unemployed+Year+Population", longley)
summary(model_1)
## Summarized Information of model
print(summary(model_1))
## Outlier Detection of model
print(outlier_detection(model_1))
## Fitted vs Residual Plot
fitted_vs_resid(model_1)
plt.show()
## QQ plot
Normal_QQ(model_1)
plt.show()
## Variance Inflation Factor
print(get_VIF(model_1))
## Analysis of Variacne
model_2 = Linear_Regression("GNP_deflator~ GNP + Unemployed", longley)
print(get_ANOVA(model_2, model_1))

## Feature Selection
print(feature_selection(model_1, method ="backward"))
