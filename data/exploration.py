"""
clean data
author: li zeng
"""

import os
os.chdir(os.path.realpath(os.curdir))

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Disable pandas warnings
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

########################################
#### LOAD TRAINING DATA ########################
########################################

train=pd.DataFrame.from_csv("./raw/train.csv")
test = pd.DataFrame.from_csv('./raw/test.csv')
train.drop(train.index[train['y']>250],inplace=True)

"""----------------------
EXPLORATION without y
----------------------""" 
# type count
sns.countplot(train.dtypes)
train.dtypes.value_counts()

# variance study
binary_df=pd.DataFrame()
zeros={}
ones={}

for col in train.select_dtypes(['int64']).columns:
    num_ones=len(train[col][train[col]==1])
    num_zeros=len(train[col][train[col]==0])
    zeros[col]=num_zeros
    ones[col]=num_ones   
binary_df['columns']=zeros.keys()
binary_df['ones']=ones.values()
binary_df['zeros']=zeros.values()
binary_df=binary_df.sort_values(by='zeros')

ind=np.arange(binary_df.shape[0])
width = 0.35
plt.figure(figsize=(6,100))
p1 = plt.barh(ind, binary_df.ones.values, width, color='yellow')
p2 = plt.barh(ind, binary_df.zeros.values, width, left=binary_df.ones.values, color="green")
plt.yticks(ind, binary_df['columns'])
plt.legend((p1[0], p2[0]), ('Zero count', 'One Count'))

# distribution of X1 - X8
var_name = "X0"
def show_var(var_name):
    temp = train[var_name].to_frame().join(train['y'])
    temp = temp.groupby(var_name).mean().sort_values(by='y')
    col_order = temp.index.values
    plt.figure(figsize=(12,10))
    sns.stripplot(x=var_name, y='y', data=train, order=col_order)
    plt.show()
    plt.figure(figsize=(12,10))
    sns.violinplot(x=var_name, y='y', data=train, order=col_order)
    plt.show()
    plt.figure(figsize=(12,10))
    sns.boxplot(x=var_name, y='y', data=train, order=col_order)
    plt.show()

show_var('X0')
show_var('X1')
show_var('X2')
show_var('X3')
show_var('X4')
show_var('X5')
show_var('X6')
show_var('X8')

"""
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

train['new_group'] = pd.Series(map(lambda x: new_group.loc[x],train['X0']),index = train.index)
g = sns.FacetGrid(train, row="new_group", margin_titles=True)
g.map(sns.distplot,'y')

# test X0
intest_only = np.setdiff1d(test['X0'].unique(),train['X0'].unique())
pd.Series(map(lambda x: np.sum(test['X0']==x),intest_only),index= intest_only)
    # total 6 samples have new X0
"""

# X0 interaction with X1-X8
def new_show_var(var_name):
    temp = train[var_name].to_frame().join(train['y'])
    temp = temp.groupby(var_name).mean().sort_values(by='y')
    col_order = temp.index.values
    g=sns.FacetGrid(train, row="X0",aspect=2)
    g.map(sns.violinplot,var_name,'y',order=col_order)
    plt.show()

new_show_var('X5')
"""----------------------
EXPLORATION with y
----------------------""" 
plt.figure(figsize=(15,10))
fig = sns.distplot(train['y'],bins=100)
plt.xlabel('Y')
plt.title('Distribution of Y variable')
fig.get_figure().savefig('y_dist.pdf')
