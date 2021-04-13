import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression
import scipy, scipy.stats
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from pandas import merge
from scipy import stats
import sklearn

import numpy as np 
import pylab 
import scipy.stats as stats
import numpy as np
import pylab
import scipy.stats as stats
from sklearn import linear_model
from pandas import merge
def data_clean(data):
    data=data[data.EXCHCD.isin([1,2,3])] # 44833839
    data=data[data.RET !='C'] ### 44814899
    data.reset_index(drop=True, inplace=True)
    data=data.fillna('NANA')
    data=data[data.RET != 'NANA'] #44709551
    data['date']=pd.to_datetime(data['date'].astype(str), format='%Y%m%d')
    data['year']=data['date'].map(lambda x: x.year)
    return data
############
def merge_data(data1, index1, data2, index2):
    data1=data1.set_index(data1[index1], drop=True)
    data2=data2.set_index(data2[index2], drop=True)
    results=merge(data1, data2, left_index=True, right_index=True, how='left', sort=True)
    results.reset_index(drop=True, inplace=True)
    return results

#################################################################
                 #camp:
def arrays(group):
    n=len(group)
    X=np.array(zip(np.ones(n), group['vwretd_rf'].astype(float))).reshape(n, 2)
    n, m=X.shape
    return X, n, m

# three factors ( created X matrix)
def arrays(group):
    n=len(group)
    X=np.array(zip(np.ones(n), group['Mkt-RF'], group['SMB'], group['HML'])).reshape(n, 4) # three factor
    n, m=X.shape
    return X, n, m


# five factors
def arrays(group):
    n=len(group)
    X=np.array(zip(np.ones(n), group['Mkt-RF'], group['SMB'], group['HML'],group['RMW'], group['CMA'])).reshape(n, 6)
    n, m=X.shape
    return X, n, m
###############################################################################
######
#regression
####################################
def linearregs((name, group)):
    X, n, m=arrays(group) # find the X matrix for each data set, n is the length of dataset, m is the variables number (degree of freedom)
    y=group['RET_rf'].reshape((n,1))
    linear = linear_model.LinearRegression()
    linear.fit(X, y)
    s=np.sum((linear.predict(X) - y.astype(float)) ** 2)/(n-(m-1)-1) ### sum square
    sd_alpha=np.sqrt(s*(np.diag(np.linalg.pinv(np.dot(X.T,X))))) # standard deviation, square root of the diagonal of variance-co-variance matrix (sigular vector decomposition)
    t_stat_alpha=linear.intercept_[0]/sd_alpha[0] #(t-statistics)
    return name[0], name[1], linear.intercept_[0], sd_alpha[0], t_stat_alpha
#################
#### substitude
                #####

data=pd.read_csv('hist.csv')
rf=pd.read_csv('Rf_All.csv')
data=data_clean(data)
all=merge_data(data, 'date', rf, 'date')
all['vwretd_rf']=all['vwretd'].astype(float)-all['rf'].astype(float) # vwretd: market
all['RET_rf']=all['RET'].astype(float)-all'rf'].astype(float) # RET: stock


##########

all=merge_data(data3, 'date', rf, 'date')
all['mkt_rf']=all['vwretd'].astype(float)-all['rf'].astype(float) # vwretd: market
all['RET_rf']=all['RET'].astype(float)-all'rf'].astype(float) # RET: stock

########################
                 
########################

data5=merge_data(factor5, 'date', rf, 'date')
data5=data_clean(data5)
all=merge_data(data5, 'date', rf, 'date')
all['mkt_rf']=all['vwretd'].astype(float)-all['rf'].astype(float) # vwretd: market
all['RET_rf']=all['RET'].astype(float)-all'rf'].astype(float) # RET: stock

######################
def arrays(group):
    n=len(group)
    X=np.array(zip(np.ones(n), group['F1'], group['F2'], group['F3'])).reshape(n, 4) # three factor
    n, m=X.shape
    return X, n, m

lsc=pd.read_csv('lsc.csv')
all_lsc=merge_data(all, 'date_x', lsc, 'date')
data=all_lsc

######################################
################
#######

cores=mp.cpu_count()
pool = Pool(processes=cores)
groups=data.groupby(['PERMNO', 'year'])
results=pool.map(linearregs, [(name, group) for name, group in groups])
results=pd.DataFrame(results)
results.rename(columns={0:'PERMNO', 1:'year', 2:'alpha',3:'sd_alpha',4:'t_stat_alpha'}, inplace=True)
results.to_csv(filename+'results.csv')
results.sort(['alpha'], axis=0, ascending=True, inplace=False)
                 
