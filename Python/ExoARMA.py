import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARMA
from statsmodels.stats.diagnostic import acorr_ljungbox

def mean_error(pred,real):
    return np.mean([pred[i]-real[i] for i in range(len(pred))])
def mean_absolute_error(pred,real):
    return np.mean([np.abs(pred[i]-real[i]) for i in range(len(pred))])
def root_mean_square_error(pred,real):
    return np.sqrt(np.mean([pow(pred[i]-real[i],2) for i in range(len(pred))]))

data = pd.read_csv("Data/VAR_data.csv")
ExchangeRateLog = data['ExchangeRateLog']
ExchangeRateLog_d_1 = data['ExchangeRateLogDifference']
delta_GDP = data['GDP']
delta_m2 = data['M2']
delta_pie = data['Inflation']
delta_r = data['ShortInterestRate']

#ARMA(1,1) 4 exgo
exog_var = np.array([delta_GDP,delta_m2,delta_pie,delta_r])
exog_var = np.transpose(exog_var)
model = ARMA(ExchangeRateLog,order=(1,1),exog=exog_var)
result = model.fit()
print(result.pvalues)
print(result.params)
print(acorr_ljungbox(result.resid)[1][-1])

#ARMA(1,1) 2 exgo
exog_var = np.array([delta_GDP,delta_pie])
exog_var = np.transpose(exog_var)
model = ARMA(ExchangeRateLog,order=(1,1),exog=exog_var)
result = model.fit()
print(result.pvalues)
print(result.params)
print(acorr_ljungbox(result.resid)[1][-1])

PredictionOutputMAE = pd.DataFrame(index=['monthly','3_month','6_month','12_month'])
PredictionOutputRMSE = pd.DataFrame(index=['monthly','3_month','6_month','12_month'])
totalData = 144
startPrediction = 100
tsValuesLog = list(ExchangeRateLog)

mae_ = []
rmse_ = []
for l in [1,3,6,12]:
    pred = [0]*(startPrediction+l-1)
    for i in range(totalData-startPrediction-l+1):
        pred.append(tsValuesLog[startPrediction-1+i])
    mae_.append(mean_absolute_error(pred[startPrediction+l-1:],tsValuesLog[startPrediction+l-1:]))
    rmse_.append(root_mean_square_error(pred[startPrediction+l-1:],tsValuesLog[startPrediction+l-1:]))
PredictionOutputMAE['RandomWalk'] = mae_
PredictionOutputRMSE['RandomWalk'] = rmse_

exog_var = np.array([delta_GDP,delta_m2,delta_pie,delta_r])
exog_var = np.transpose(exog_var)
mae_ = []
rmse_ = []
for l in [1,3,6,12]:
    pred = [0]*(startPrediction+l-1)
    for i in range(totalData-startPrediction-l+1):
        history = tsValuesLog[i:i+startPrediction]
        exog_history = exog_var[i:i+startPrediction]
        model = ARMA(endog=history,order=(1,1),exog=exog_history)
        result = model.fit()
        prediction = result.predict(start = startPrediction, end = startPrediction+l-1,exog=exog_var[startPrediction:startPrediction+l])
        pred.append(prediction[l-1])
    mae_.append(mean_absolute_error(pred[startPrediction+l-1:],tsValuesLog[startPrediction+l-1:]))
    rmse_.append(root_mean_square_error(pred[startPrediction+l-1:],tsValuesLog[startPrediction+l-1:]))
PredictionOutputMAE['ARMA_1_1+Exo_4'] = mae_
PredictionOutputRMSE['ARMA_1_1+Exo_4'] = rmse_

exog_var = np.array([delta_GDP,delta_pie])
exog_var = np.transpose(exog_var)
mae_ = []
rmse_ = []
for l in [1,3,6,12]:
    pred = [0]*(startPrediction+l-1)
    for i in range(totalData-startPrediction-l+1):
        history = tsValuesLog[i:i+startPrediction]
        exog_history = exog_var[i:i+startPrediction]
        model = ARMA(endog=history,order=(1,1),exog=exog_history)
        result = model.fit()
        prediction = result.predict(start = startPrediction, end = startPrediction+l-1,exog=exog_var[startPrediction:startPrediction+l])
        pred.append(prediction[l-1])
    mae_.append(mean_absolute_error(pred[startPrediction+l-1:],tsValuesLog[startPrediction+l-1:]))
    rmse_.append(root_mean_square_error(pred[startPrediction+l-1:],tsValuesLog[startPrediction+l-1:]))
PredictionOutputMAE['ARMA_1_1+Exo_2'] = mae_
PredictionOutputRMSE['ARMA_1_1+Exo_2'] = rmse_

print(PredictionOutputMAE)
print(PredictionOutputRMSE)
