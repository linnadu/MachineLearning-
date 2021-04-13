from pandas import merge
data_95=pd.read_csv("/Users/dulinna/Dropbox/Cross-Section_of_Alpha/CRSP_Daily_9803.csv")
data_14=pd.read_csv("/Users/dulinna/Dropbox/Cross-Section of Alpha/CRSP_Daily_2005_2014.csv")
data_3factor=pd.read_csv("/Users/dulinna/Dropbox/Cross-Section of Alpha/F-F_Research_Data_Factors_daily.CSV")
data_5factor=pd.read_csv("/Users/dulinna/Dropbox/Cross-Section of Alpha/F-F_Research_Data_5_Factors_2x3_daily.CSV")
rf=pd.read_csv("/Users/dulinna/Dropbox/Cross-Section of Alpha/Rf_All.csv")
rf['date']=pd.to_datetime(rf['date'].astype(str), format='%Y%m%d')

data_3factor=pd.read_csv("/Users/dulinna/Dropbox/Cross-Section of Alpha/FactorsDailyFF3.csv")
data=pd.read_csv("CRSP_Daily_2005_2014.csv")
data=pd.read_csv("CRSP_Daily_9803.csv")

q=datetime.datetime.strptime('03-Jan-05',"%d-%b-%y")
data_3factor['date']=data_3factor['date'].map(lambda x: str(datetime.datetime.strptime(x,"%d-%b-%y"))[:10])

###date=datetime.datetime.fromtimestamp(float(date)).strftime('%Y-%m-%d')
data['date']=pd.to_datetime(data['date'].astype(str), format='%Y%m%d')

data_14.to_csv("/Users/dulinna/Dropbox/Cross-Section of Alpha/CRSP_Daily_2005_2014_clean.csv")
data_14=pd.read_csv("/Users/dulinna/Dropbox/Cross-Section of Alpha/CRSP_Daily_2005_2014_clean.csv")


data_10051=data_2[data_2['PERMNO']==10051]
data_10051_2013=data_10051[data_10051['year']==2013]

all=merge_data(data_10051_2013, 'date_x', rf, 'date')
all['vwretd_rf']=(all['vwretd'].astype(float)-all['rf'].astype(float)) # vwretd: market
all['RET_rf']=(all['RET'].astype(float)-all['rf'].astype(float))# RET: stock
all=merge_data(all, 'date_x', data_3factor, 'date')

slope, intercept, r_value, p_value, std_err = stats.linregress(group['vwretd'].astype(float),group['RET'].astype(float))

slope, intercept, r_value, p_value, std_err = stats.linregress(group['vwretd_rf'].astype(float),group['RET_rf'].astype(float))


n=len(group)
X=np.array(zip(np.ones(n), group['vwretd_rf'].astype(float))).reshape(n, 2)
y=group['RET_rf'].reshape((n,1))


group_13977=group
n, m=X.shape
import numpy as np
import matplotlib.pyplot as plt

predic_python=linear.predict(X)

#plt.plot(group['vwretd'],group['RET'])

fig=plt.figure()
ax1=fig.add_subplot(111)
ax1.scatter(group['vwretd'], group['RET']) 
ax1.plot(group['vwretd'],predic_python, 'bs', group['vwretd'],predic_stata, 'g^')
ax1.plot(group['vwretd'],predic_stata, label='stata')

###############
data_3factor['date']=pd.to_datetime(data_3factor['date'].astype(str), format='%Y%m%d')
data_3factor['year']=data_3factor['date'].map(lambda x: x.year)

factor3=merge_data(group, 'date_x', data_3factor, 'date')
