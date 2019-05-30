import pandas as pd
import numpy as np
import math
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot as plt
from statsmodels.tsa.x13 import x13_arima_analysis
import datetime

#You can change dataset here
dataset = 'EXUSEU'
data = pd.read_csv('Data/'+dataset+".csv")
tsValues = list(data[dataset])
startPrediction = 100
totalData = len(tsValues)
#log data
tsValuesLog = [math.log(ele) for ele in tsValues]
#first difference
tsValuesLog_d_1 = [0]
tsValuesLog_d_1 += [tsValuesLog[i+1]-tsValuesLog[i] for i in range(totalData-1)]

#See the picture
plt.figure(figsize=(20,5))
plt.plot(tsValuesLog)
plt.title("US-"+dataset+" Exchange Rate",fontsize=20)
plt.show()

#ADF test
print(adfuller(tsValuesLog,regression='c'))
print(adfuller(tsValuesLog_d_1,regression='c'))

#ACF-PACF
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(tsValuesLog, lags=30, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(tsValuesLog, lags=40, ax=ax2)
fig.show()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(tsValuesLog_d_1, lags=30, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(tsValuesLog_d_1, lags=40, ax=ax2)
fig.show()

#Try seasonal decomposition
decomp = seasonal_decompose(tsValuesLog,freq=12)
fig = plt.figure(figsize=(20,5))
fig = decomp.plot()
fig.show()

#Try x13 to make seasonal adjusted ts
date = [datetime.datetime.strptime(ele,'%Y-%m-%d') for ele in data['observation_date']]
ts = pd.DataFrame(index = date)
ts['tsValuesLog'] = tsValuesLog
model = x13_arima_analysis(ts)
tsValuesLog_seasadj = list(model.seasadj)
plt.figure(figsize=(20,5))
plt.plot(model.seasadj)
plt.title("Seansonal Adjusted Exchange Rate (log)",fontsize = 20)
plt.show()

