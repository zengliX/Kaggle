"""
clean data
"""

import os
os.chdir('/Users/lizeng/Google Drive/Kaggle/data')

import pandas as pd

########################################
#### LOAD TRAINING DATA ########################
########################################

train_data = pd.DataFrame.from_csv('./raw/train.csv')
response = train_data['y']
