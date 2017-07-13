"""
clean data
author: li zeng
"""

import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

#Disable pandas warnings
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import mannwhitneyu

########################################
#### LOAD TRAINING DATA ########################
########################################

train=pd.DataFrame.from_csv("./raw/train.csv")
test = pd.DataFrame.from_csv('./raw/test.csv')
train.drop(train.index[train['y']>250],inplace=True)


"""----------------------
EXPLORATION without y
----------------------"""

# duplicated rows
dup_rows = train.iloc[:,1:-1].duplicated(keep=False)
temp = train[dup_rows].sort_values(by=['X0','X1','X2','X3','X4','X5','X6','X8'])


# pivot_table
train.groupby(['X0','X1'])['y'].mean()
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

# X0 interaction with X1-X8
def new_show_var(var_name):
    temp = train[var_name].to_frame().join(train['y'])
    temp = temp.groupby(var_name).mean().sort_values(by='y')
    col_order = temp.index.values
    g=sns.FacetGrid(train, row="X0",aspect=2)
    g.map(sns.violinplot,var_name,'y',order=col_order)
    plt.show()

new_show_var('X5')


# X5 vs ID
train['ID']=train.index
sns.boxplot(x='X5',y='ID',data=train)

# X0 vs X10-X385
train2 = pd.DataFrame.from_csv("./cleaned3/train.csv")
del train2['y']
train2['y'] = train['y']
train2['X0'] = train['X0']
train2.sort_values(by=['X0','y'],inplace=True)

# add residual
X0_cols = list(filter(lambda x: 'X0'+'_' in x, train2.columns))
linear_fit = LinearRegression(fit_intercept=False)
linear_fit.fit(train2[X0_cols],train2['y'])
y_linear_train = linear_fit.predict(train2[X0_cols]) # linear pred on train
res = train2['y'] - y_linear_train # residual
train2['res'] = res
train2.sort_values(by=['res'],inplace=True)

# remove duplicated columns
bin_cols = [x for x in train2.columns if not '_' in x][:-3]
temp = train2[bin_cols].T.duplicated()
train2.drop(temp.index[temp],axis=1,inplace=True)


def show_x0_group(v):
    plt.figure(figsize=(15,12))
    sns.heatmap(train2.loc[train2['X0']==v,out.index],xticklabels=False, yticklabels=True)
    plt.show()
    train2.loc[train2['X0']=='ak',['y','res']]
    #sns.heatmap(train2.loc[train2['X0']=='ak',train2.columns[:-2]],xticklabels=False, yticklabels=True)
    sns.distplot(train2.loc[train2['X0']==v,'res'],bins=40)
    plt.show()
    print(train2.loc[train2['X0']==v,['X0','y','res'] ])

show_x0_group('g')

plt.figure(figsize=(15,12))
sns.heatmap(train.loc[train['new_group']=='C',train.columns[9:-2]],xticklabels=False, yticklabels=False)
sns.distplot(train.loc[train['new_group']=='C','y'])


# mean for each binary col
bin_cols = [x for x in train2.columns if not '_' in x][:-3]
def f(x):
    out=mannwhitneyu(res[x==1],res[x==0])
    return out.pvalue

out = train2[bin_cols].apply(f)
out.sort_values(inplace=True)
pd.Series(out.index[out<0.01]).to_csv('sele_cols.csv')
plt.scatter(range(len(out)),out.values)



# distribution of each binary col
def X_distr(v):
    n = sum(train2[v])
    print('number of 1s:',n)
    sns.distplot(train2.loc[train2[v]==1,'res'])
    plt.show()
    sns.distplot(train2.loc[train2[v]==0,'res'])
    plt.show()
    print(train2.loc[:,['X0','res','y',v]].groupby(v).mean())
    print(train2.loc[:,['X0','res','y',v]].groupby(v).size())

X_distr('X314')


# correlation between binary cols
corr_mat = train2[out.index].corr()
sns.heatmap(corr_mat)
sum((corr_mat>0.95).values.reshape(-1) )
# cluster by binary values
from sklearn.cluster import KMeans

temp = train2.groupby('X0')
sort_y = temp.mean()['y'].sort_values()
x0_centers = temp.mean().loc[sort_y.index,bin_cols]
plt.figure(figsize=(15,12))
sns.heatmap(x0_centers)

km = KMeans(n_clusters = x0_centers.shape[0],init = x0_centers,max_iter=10).fit(train2[bin_cols])
new_x0 = pd.Series(x0_centers.index[km.labels_],index=train2.index)
new_x0.rename('new_x0',inplace=True)

# new x0 from clustering results
def show_x0_group2(v):
    plt.figure(figsize=(15,12))
    sns.heatmap(train2.loc[new_x0==v,out.index],xticklabels=False, yticklabels=True)
    plt.show()
    sns.distplot(train2.loc[new_x0==v,'res'],bins=40)
    plt.show()
    print(train2.loc[new_x0==v,['X0','y'] ])

show_x0_group2('ay')

def check_new_group(v):
    temp = train2.loc[train2.X0==v,'y']
    inds = temp[temp > temp.mean() + 1.96*temp.std()].index
    print(pd.concat([train2.loc[inds,'X0'],new_x0[inds]],axis=1))
    print('old:')
    sns.distplot(train2.loc[train2['X0']==v,'y'],bins=40)
    plt.show()
    print('new:')
    sns.distplot(train2.loc[new_x0==v,'y'],bins=40)
    plt.show()

check_new_group('ay')
"""----------------------
EXPLORATION with y
----------------------"""

# distribution of y
plt.figure(figsize=(15,10))
fig = sns.distplot(train['y'],bins=100)
plt.xlabel('Y')
plt.title('Distribution of Y variable')
fig.get_figure().savefig('y_dist.pdf')


# y vs row sum
cars = ['X'+str(x) for x in range(9)]+['y']
def f(x):
    return any([(z == x) for z in cars ])
tests_cols = [x for x in train.columns if not f(x)]

Nt = train[tests_cols].sum(axis=1)
train['Nt'] = Nt
plt.scatter(Nt,y)


# correlation component analysis
y = train['y']
train.drop(['y','new_group'],axis=1,inplace=True)
pls = PLSRegression(n_components=10,scale=False)
pls.fit(train[tests_cols],res)
rotations = pls.x_rotations_
plt.plot(rotations[:,0])
plt.scatter(train[tests_cols].dot(rotations[:,2]),res)
