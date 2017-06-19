"""
final data preprocessing script
author: li zeng
"""

import pandas as pd
import numpy as np

""" -------------------------
Parameters
percent_cut: select variables that both group frequency > percent_cut
-------------------------"""
percent_cut = 0.02


""" -------------------------
CLEAN DATA
-------------------------"""


TRAIN = pd.DataFrame.from_csv('./raw/train.csv')
TEST = pd.DataFrame.from_csv('./raw/test.csv')
tr = pd.concat([TRAIN,TEST],axis=0,join='outer')

# change categorical to dummy variables
tr = pd.get_dummies(tr,columns = tr.select_dtypes(['object']).columns)
    
# remove columns with low variance
to_del = []
for col in tr.columns:
    if tr[col].dtype == 'int64':
        zero_freq = np.sum(tr[col]==0)/len(tr)
        if zero_freq > 1-percent_cut or zero_freq < percent_cut:
            to_del.append(col)
tr = tr.drop(to_del,axis=1)
tr = tr[np.logical_or(tr['y']<175, np.isnan(tr['y'])) ]
    
TRAIN = tr.loc[np.intersect1d(tr.index,TRAIN.index)]
TEST= tr.loc[TEST.index]
    
TRAIN.to_csv('./cleaned/train.csv')
TEST.to_csv('./cleaned/test.csv')
