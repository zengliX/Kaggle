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
import math

#Disable pandas warnings
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

########################################
#### LOAD TRAINING DATA ########################
########################################

train=pd.DataFrame.from_csv("./raw/train.csv")

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

"""----------------------
EXPLORATION with y
----------------------""" 
