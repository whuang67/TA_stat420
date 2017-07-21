# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 21:12:51 2017

@author: whuang67
"""

import numpy as np
import pandas as pd
import itertools
import os

os.chdir("C:/Users/whuang67/downloads")
for i in range(0, 15):
    if i == 0:
        a = "kdd15-p9"
    else:
        a = "kdd15-p"+str(i)+"9"
    print a

dat1 = ["John", "st","LI"]
dat2 = [100,2,300]
dat3 = ['a', 'b']
dat = list(itertools.product(dat1, dat2, dat3))
pd.DataFrame(dat, columns = ['1', 'a', 'c'])

df = pd.read_excel()