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


train = TRAIN = pd.DataFrame.from_csv('./raw/train.csv')
TEST = pd.DataFrame.from_csv('./raw/test.csv')
tr = pd.concat([TRAIN,TEST],axis=0,join='outer')

# separate X0 into 4 groups
var_name= 'X0'
temp = train[var_name].to_frame().join(train['y'])
temp = temp.groupby(var_name).mean().sort_values(by='y')
new_group = pd.Series(range(len(temp.y)),index = temp.index)
for i in new_group.index:
    if temp.loc[i].values < 80:
        new_group.loc[i] = 'A';
    elif temp.loc[i].values < 96:
        new_group.loc[i] = 'B'
    elif temp.loc[i].values < 108:
        new_group.loc[i] = 'C'
    else:
        new_group.loc[i] = 'D'


def f(x):
    if x in new_group.index:
        return new_group.loc[x]
    return 'E'
tr['new_group'] = pd.Series(map(f,tr['X0']),index = tr.index)


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
    
TRAIN.to_csv('./cleaned2/train.csv')
TEST.to_csv('./cleaned2/test.csv')

