"""
final data preprocessing script
author: li zeng
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
import matplotlib.pyplot as plt

""" -------------------------
Parameters
percent_cut: select variables that both group frequency > percent_cut
-------------------------"""
percent_cut = 0.001


""" -------------------------
CLEAN DATA
-------------------------"""

train = TRAIN = pd.DataFrame.from_csv('./raw/train.csv')
TEST = pd.DataFrame.from_csv('./raw/test.csv')
tr = pd.concat([TRAIN,TEST],axis=0,join='outer')

# change categorical to dummy variables
tr = pd.get_dummies(tr,columns = tr.select_dtypes(['object']).columns)

# perform kmeans to cluster columns
bin_cols = [x for x in tr.columns if not '_' in x][:-1]
temp =  tr[bin_cols].T


def eblow(df, n):
    kMeansVar = [KMeans(n_clusters=k).fit(df.values) for k in np.arange(10,n,2)]
    centroids = [X.cluster_centers_ for X in kMeansVar]
    k_euclid = [cdist(df.values, cent) for cent in centroids]
    dist = [np.min(ke, axis=1) for ke in k_euclid]
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(df.values)**2)/df.values.shape[0]
    bss = tss - wcss
    plt.plot(np.arange(10, n,2),bss/tss)
    plt.show()

eblow(temp,60)

km = KMeans(n_clusters = 50).fit(temp)
km_cols = pd.DataFrame(km.cluster_centers_.T,index = tr.index,columns=['center_'+str(x) for x in range(50)])

# add kmeans columns to tr

tr = tr.join(km_cols)

tr = tr[np.logical_or(tr['y']<175, np.isnan(tr['y'])) ]

TRAIN = tr.loc[np.intersect1d(tr.index,TRAIN.index)]
TEST= tr.loc[TEST.index]

TRAIN.to_csv('./cleaned4/train.csv')
TEST.to_csv('./cleaned4/test.csv')
